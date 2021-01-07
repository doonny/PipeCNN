/*
 * ------------------------------------------------------
 *
 *   PipeCNN: An OpenCL-Based FPGA Accelerator for CNNs
 *
 * ------------------------------------------------------
 * Filename:
 *   - conv_pipe.cl
 *
 * Author(s):
 *   - Dong Wang, wangdong@m.bjtu.edu.cn
 *
 * History:
 *   - v1.3 Win-Buffer-Based Implementation
 * ------------------------------------
 *
 *   Copyright (C) 2019, Institute of Information Science,
 *   Beijing Jiaotong University. All rights reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 */

// The following macros are used for debug
//#define DEBUG_MEMRD
//#define DEBUG_CONV
//#define DEBUG_BN
//#define DEBUG_POOL
//#define DEBUG_MEMWR
//#define DEBUG_LRN
//#define DEBUG_LRN_OUT

#include "hw_param.h"


// parallel MAC units including (VEC_SIZE-1) multipliers
MACTYPE mac(lane_data input, lane_data weights)
{
#pragma HLS inline

    MACTYPE output = MASK_MULT & CZERO;

    for(int i=0; i<VEC_SIZE; i++) {
		#pragma HLS unroll
        output += input.data[i]*weights.data[i];
    }
    return output;
}


extern "C" {
void coreConv(
    // Params Ports
    uint  output_num,
    uint  conv_loop_cnt,
    uint  contol, //[0]-> relu  [1]->bypass pooling,for ResNet [0]->bn.[1]->wr(fc)
    char  frac_w,
    char  frac_din,
    char  frac_dout,
	hls::stream<k2k_data_xlane>     &bias_in,
	hls::stream<k2k_data_vecxlane>  &weight_in,
	hls::stream<k2k_data_vecxlane>  &data_in,
#ifdef RESNET
	hls::stream<k2k_data_xlane>     &bypass_out,
#endif
	hls::stream<k2k_data_xlane>     &conv_out
)
{
    channel_vec mac_data;
    channel_vec mac_weight;
    channel_scal bias_ch_out;
    channel_scal conv_ch_in;
	k2k_data_xlane bias_in_tmp;
	k2k_data_vecxlane data_in_tmp;
	k2k_data_vecxlane weight_in_tmp;
	k2k_data_xlane conv_out_tmp;

    DPTYPE  bias[LANE_NUM];
    MACTYPE conv_acc[LANE_NUM];
    MACTYPE lane_accum[LANE_NUM];
    MACTYPE accum_piped[LANE_NUM][PIPE_DEPTH];
    #pragma HLS ARRAY_PARTITION variable=accum_piped dim=0 complete
    MACTYPE conv_sign_exten[LANE_NUM];
    MACTYPE conv_with_rnd_bit[LANE_NUM];
    MACTYPE conv_sum_bias[LANE_NUM];
    DPTYPE  conv_final[LANE_NUM];

	int conv_inner_cnt = 0;// loop index within one conv kernel, for example a 3*3*1024 conv kernel, the conv_inner_cnt should between 0 and 3*3*1024/VEC_SIZE

	// conv_loop_cnt iterations generate one 1x1xLANE_NUM output pixels, 
	for(unsigned int k=0; k<output_num*conv_loop_cnt; k++){
	//The "OutputNum" and "ConvLppo", loops are merged to improve pipeline efficiency.
    //OutputNum:for(unsigned int k=0; k<output_num; k++){
   		if(conv_inner_cnt == 0){//starting a new conv kernel    
		bias_in_tmp = bias_in.read();
		for(unsigned char ll=0; ll<LANE_NUM; ll++){
			#pragma HLS unroll
			bias_ch_out.lane[ll] = bias_in_tmp.data(ll*DP_WIDTH+DP_WIDTH-1, ll*DP_WIDTH);
		}
        //bias_ch_read_pipe_block(bias_ch_out);

		for(unsigned char ll=0; ll<LANE_NUM; ll++){
			#pragma HLS unroll
			bias[ll] = bias_ch_out.lane[ll]; // pass to reg, avoid compile error

			// initialize the deep pipelined registers which store PIPE_DEPTH copys of partial results
			for(unsigned int p=0; p<PIPE_DEPTH; p++){
				#pragma HLS unroll
				accum_piped[ll][p] = MASK_ACCUM & CZERO;
			}
		}
        }

        //ConvLppo:for(int j=0; j<conv_loop_cnt; j++){//1x1xLANE_NUM conv output will be generated ones this loop is finished
            // load data and weights for each lane
			data_in_tmp = data_in.read();
			weight_in_tmp = weight_in.read();
            for(unsigned char ll=0; ll<LANE_NUM; ll++){
				#pragma HLS unroll
				for(unsigned char vv=0; vv<VEC_SIZE; vv++){ // copy data_vec to each lane
					#pragma HLS unroll
					mac_data.lane[ll].data[vv] = data_in_tmp.data(VEC_SIZE*DP_WIDTH*ll+(vv*DP_WIDTH+DP_WIDTH-1), VEC_SIZE*DP_WIDTH*ll+(vv*DP_WIDTH));
					mac_weight.lane[ll].data[vv] = weight_in_tmp.data(VEC_SIZE*DP_WIDTH*ll+(vv*DP_WIDTH+DP_WIDTH-1), VEC_SIZE*DP_WIDTH*ll+(vv*DP_WIDTH));
				}
            }
            //data_read_pipe_block(mac_data);
            //weight_read_pipe_block(mac_weight);

			// add results from all lanes
			// accumulate with the last copy
            for(unsigned char ll=0; ll<LANE_NUM; ll++){
				#pragma HLS unroll

                lane_accum[ll] = (MASK_ACCUM & accum_piped[ll][PIPE_DEPTH-1]) + (MASK_MULT & mac(mac_data.lane[ll], mac_weight.lane[ll]));

                // Shift the pipelined registers backwards
                for(unsigned int p=PIPE_DEPTH-1; p>0; p-- ){
					#pragma HLS unroll
                    accum_piped[ll][p] = MASK_ACCUM & accum_piped[ll][p-1];
                }

                // update the first copy
                accum_piped[ll][0] = MASK_ACCUM & lane_accum[ll];

                #ifdef DEBUG_CONV
                //if(ll==0 && k==0){
                //	printf("dot_cnt=%d data=%f weight=%f (loop=%d, lane= %d, vec=0)\n", k, (float)mac_data.lane[ll].data[0], (float)mac_weight.lane[ll].data[0], j, ll);
                //}
                #endif
            }
        //}// end of ConvLppo

	if(conv_inner_cnt == conv_loop_cnt-1){//One ConvLoop is finished, and 1x1xLANE_NUM conv output is generated. Then the bias and rounding operation is performed.
        for(unsigned char ll=0; ll<LANE_NUM; ll++){
			#pragma HLS unroll
			conv_acc[ll] = CZERO;
            // accumulate all the partial results
            for(unsigned i=0; i<PIPE_DEPTH; i++){
				#pragma HLS unroll
                conv_acc[ll] += accum_piped[ll][i];
            }

			// round and truncate the results to the output precision
			// note: ((frac_w+frac_din)-frac_dout)) should be checked by host to be a positive number
            if(conv_acc[ll]>=0)
                conv_sign_exten[ll] = 0x00;
            else
                conv_sign_exten[ll] = ~(0xFFFFFFFF>>(frac_w+frac_din-frac_dout-1)); // ">>" is logic shift, then perform sign extension manually

			 // First, perform sign extension and the 1st-step rounding before sum with bias
            conv_with_rnd_bit[ll] = (conv_sign_exten[ll] | (conv_acc[ll]>>(frac_w+frac_din-frac_dout-1))) + 0x01;

			// Second, deal with Overflow and Underflow cases and the 2nd rounding after sum with bias
            if(conv_with_rnd_bit[ll]>=256)
                conv_sum_bias[ll] = MASK9B & 0xFF; //=255
            else if(conv_with_rnd_bit[ll]<-256)
                conv_sum_bias[ll] = MASK9B & 0x100; //=-256
            else
			    // clear 1st-step rounding bit by using MASK9B
				// then sum with bias and perform 2nd-step rounding
				// note: (frac_w-frac_dout-1) should be checked by host to be a positive number
                conv_sum_bias[ll] = (MASK9B & conv_with_rnd_bit[ll])+(bias[ll]>>(frac_w-frac_dout-1))+0x01;

            // final truncation
            conv_final[ll] = MASK8B & (conv_sum_bias[ll]>>0x01);  // remove the last rounding bit

#ifdef RESNET
			conv_ch_in.lane[ll]= conv_final[ll];
		}
		//BatchNorm
		if(contol==0){
		    for(unsigned char ll=0; ll<LANE_NUM; ll++){
			    #pragma HLS unroll
			    conv_out_tmp.data(ll*DP_WIDTH+DP_WIDTH-1, ll*DP_WIDTH) = conv_ch_in.lane[ll];
	    	}
		    conv_out.write(conv_out_tmp);
            //conv_ch_write_pipe_block(conv_ch_in);
        }
		else{
		    for(unsigned char ll=0; ll<LANE_NUM; ll++){
			    #pragma HLS unroll
			    conv_out_tmp.data(ll*DP_WIDTH+DP_WIDTH-1, ll*DP_WIDTH) = conv_ch_in.lane[ll];
		    }
		    bypass_out.write(conv_out_tmp);
            //bypass_bn_ch_write_pipe_block(conv_ch_in);
        }
#else

            // Relu operation
            if((contol&0x01)==0x01){
                if((conv_final[ll]&MASKSIGN)==MASKSIGN) // MSB is sign bit
                    conv_ch_in.lane[ll] = 0;
                else
                    conv_ch_in.lane[ll] = conv_final[ll];
            }
            else
                conv_ch_in.lane[ll] = conv_final[ll];
                
            #ifdef DEBUG_CONV
            if(ll==0 && k==0)
                printf("dot_cnt=%d sum=%f rnd=%f sum_bias=%f final=%f (bias=%f)\n\n", k, (float)conv_acc[ll], (float)conv_with_rnd_bit[ll], (float)conv_sum_bias[ll], (float)conv_final[ll], (float)bias[ll]);
            #endif

        }

		for(unsigned char ll=0; ll<LANE_NUM; ll++){
			#pragma HLS unroll
			conv_out_tmp.data(ll*DP_WIDTH+DP_WIDTH-1, ll*DP_WIDTH) = conv_ch_in.lane[ll];
		}
		conv_out.write(conv_out_tmp);
        //conv_ch_write_pipe_block(conv_ch_in);
#endif
		conv_inner_cnt = 0;//one conv kernel is finished, reset counter
	}//Bias and rounding operation is finished.
	else{//One conv kernel is not finished, continued.
		conv_inner_cnt++;
	}
    //}// end of output loop
    }// end of merged loop
    //printf("Kernel coreConv lanched !!!\n");
}
}