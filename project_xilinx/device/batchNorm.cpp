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

extern "C" {
void batchNorm(
				uint dim1xdim2,
				uint input_num,//dim1*dim2*dim3/LANE_NUM
				uint  contol, //[0]-> relu  [1]->bypass pooling
				//char  frac_dout,
				float frac2float,//conv out conver to float
				float frac2char,//bn out conver to char
				const channel_scal_float     *mean,
				const channel_scal_float     *var,
				const channel_scal_float     *alpha,
				const channel_scal_float     *beta,
	            hls::stream<k2k_data_xlane>  &conv_in,
	            hls::stream<k2k_data_xlane>  &bn_out
                )
{
#pragma HLS INTERFACE m_axi port = mean  offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = var  offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = alpha  offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = beta  offset = slave bundle = gmem3

	channel_scal conv_ch_out;
	channel_scal batchNorm_final;
	channel_scal bn_ch_in;
	float bn_in;
	float bn_out_fp;
	float sc_out;
	channel_scal_float mean_ch;
	channel_scal_float var_ch;
	channel_scal_float alpha_ch;
	channel_scal_float beta_ch;

	k2k_data_xlane conv_in_tmp;
	k2k_data_xlane bn_out_tmp;

	unsigned int iter=0;
	unsigned int j=dim1xdim2;

	DPTYPE out_final;
	float out_conver;
	for(unsigned int k=0; k<input_num; k++,j++){
		conv_in_tmp = conv_in.read();
		for(unsigned char ll=0; ll<LANE_NUM; ll++){
			#pragma HLS unroll
			conv_ch_out.lane[ll] = conv_in_tmp.data(ll*DP_WIDTH+DP_WIDTH-1, ll*DP_WIDTH);
		}
        //conv_ch_read_pipe_block(conv_ch_out);
		if(j==dim1xdim2)
		{
			mean_ch = mean[iter];
			var_ch = var[iter];
			alpha_ch = alpha[iter];
			beta_ch = beta[iter];
			iter=iter+1;
			j=0;
		}
		
		// #pragma unroll
		for(unsigned char ll=0; ll<LANE_NUM;ll++){
			// ll=part*LANE_NUM_DIV_PART_NUM+part_ll;
			// Convert DPTYPE fixed-point to float
			// Input data has "frac_dout" fractional bits
			// bn_in = convert_float(conv_ch_out.lane[ll])*pow(2,-frac_dout);
			// bn_in = convert_float(conv_ch_out.lane[ll])*frac2float;
            bn_in = ((float)conv_ch_out.lane[ll])*frac2float;
			// top(:,:,n)=(bottom(:,:,n)-mean(n))/(variance(n).^0.5);
			bn_out_fp=(bn_in-mean_ch.lane[ll])*var_ch.lane[ll];
			//top(:,:,n)=bottom(:,:,n)*alpha(n)+beta(n);
			sc_out=bn_out_fp*alpha_ch.lane[ll]+beta_ch.lane[ll];

			// Convert float to DPTYPE fixed-point
			// out_conver=sc_out*pow(2,frac_dout);
			out_conver=sc_out*frac2char;
			if(out_conver>=0)
				out_conver=out_conver+0.5;
			else
				out_conver=out_conver-0.5;

			if(out_conver>127)
				out_conver=127;
			else if(out_conver<-128)
				out_conver=-128;
			// batchNorm_final.lane[ll]=convert_char_rtz(out_conver);//Round towards zero
            batchNorm_final.lane[ll]=(char)out_conver;//Round towards zero
			// Relu operation
			if((contol&0x01)==0x01){
				if((batchNorm_final.lane[ll]&MASKSIGN)==MASKSIGN) // MSB is sign bit
					bn_ch_in.lane[ll] = 0;
				else
					bn_ch_in.lane[ll] = batchNorm_final.lane[ll];
			}
			else
				bn_ch_in.lane[ll] = batchNorm_final.lane[ll];
#ifdef DEBUG_BN
			printf("ll=%d,conv_ch_out=%d,bn_in=%f,sc_out=%f,batchNorm_final.lane[ll]=%d,bn_ch_in.lane[ll]=%d\n",ll,conv_ch_out.lane[ll],bn_in,sc_out,batchNorm_final.lane[ll],bn_ch_in.lane[ll]);
#endif
		}

	for(unsigned char ll=0; ll<LANE_NUM; ll++){
		#pragma HLS unroll
		bn_out_tmp.data(ll*DP_WIDTH+DP_WIDTH-1, ll*DP_WIDTH) = bn_ch_in.lane[ll];
	}
    bn_out.write(bn_out_tmp);
	//batchNorm_ch_write_pipe_block(bn_ch_in);
	//printf("Write channel item-%d is written in channel %d...\n", k, ll);
	}
	//printf("Kernel batchNorm lanched !!!\n");
}
}