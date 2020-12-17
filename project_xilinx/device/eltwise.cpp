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
void eltwise(
			uint  input_num,//dim1*dim2*dim3/VEC_SIZE
			uchar pool_on,//only avgpool used
			//uchar pool_size,  // by now, only pooling size is 7
			uchar  conv_x,
			uint  conv_xy,
			uchar  stride,
			float divisor,	  //1/pool_size^2
			float in1_frac,
			float in2_frac,//conver the input frac to output frac
			// float out_conver2char,
			const lane_data *bottom_1,
			const lane_data *bottom_2,
			lane_data *top
			)
{
#pragma HLS INTERFACE m_axi port = bottom_1  offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = bottom_2  offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = top  offset = slave bundle = gmem2

	lane_data data_out;
	float sum;
	// float sum[VEC_SIZE];
	float sumAvg[VEC_SIZE];
	float out;
	//uchar pool_size_num;  //pool_size^2
	//pool_size_num=pool_size*pool_size;
	uchar conv_itm_x=0;
	uint  conv_itm_xyz=0;
	uint xyz_offset=0;
	uint xy_offset=0;
	uint ptr=0;
	uint outnum=0;//if have avgpool the out ptr
	float avgPoolSum[VEC_SIZE];
	float avgPoolBuf[VEC_SIZE][ELT_PIPE_DEPTH];
	//__attribute__((opencl_unroll_hint))
	for(unsigned char vv=0; vv<VEC_SIZE; vv++){
		avgPoolSum[vv]=0;
		//__attribute__((opencl_unroll_hint))
		for(unsigned char pp=0; pp<ELT_PIPE_DEPTH; pp++){
			avgPoolBuf[vv][pp]=0;
		}
	}

	for(unsigned int j=0;j<input_num;j++)
	{
		//__attribute__((opencl_unroll_hint))
		for(unsigned char vv=0; vv<VEC_SIZE; vv++){
			sum=((float)bottom_1[j].data[vv])*in1_frac+((float)bottom_2[j].data[vv])*in2_frac;
			//sum=bottom_1[j].data[vv]*in1_frac+bottom_2[j].data[vv]*in2_frac;
			// relu
			if(sum<0)
				sum=0;
			if(pool_on!=3){
				sum=sum+0.5;
				// //overflow
				if(sum>127)
					sum=127;
				// data_out.data[vv]=convert_char_rtz(sum);//Round towards zero
                data_out.data[vv]=(char)(sum);
			}
			else{
				sumAvg[vv]=sum+avgPoolBuf[vv][ELT_PIPE_DEPTH-1];
				//__attribute__((opencl_unroll_hint))
				for(uchar p=ELT_PIPE_DEPTH-1; p>0; p-- ){
					avgPoolBuf[vv][p] = avgPoolBuf[vv][p-1];
				}
				avgPoolBuf[vv][0]=sumAvg[vv];
			}

		}
		if(pool_on==0){
			top[j]=data_out;
		}
		else if(pool_on==2){//stride pool
			conv_itm_xyz=xyz_offset+xy_offset+conv_itm_x;
			if(conv_itm_xyz==j){
				top[outnum]=data_out;
				outnum++;
				conv_itm_x=conv_itm_x+stride;
				if(conv_itm_x>=conv_x){
					conv_itm_x=0;
					xy_offset+=stride*conv_x;
					if(xy_offset>=conv_xy){
						xy_offset=0;
						xyz_offset+=conv_xy;
					}
				}
			}

		}
		else if(pool_on==3){//avgpool
			ptr=ptr+1;
			if(ptr==AVGPOOL_SIZE)
			{
				ptr=0;
				//__attribute__((opencl_unroll_hint))
				for(unsigned char vv=0; vv<VEC_SIZE; vv++){
					//__attribute__((opencl_unroll_hint))
					for(unsigned i=0; i<ELT_PIPE_DEPTH; i++){
						avgPoolSum[vv] += avgPoolBuf[vv][i];
						avgPoolBuf[vv][i]=0;
					}
					out=avgPoolSum[vv]*divisor+0.5;
					//overflow,because of relu no <0 value here
					if(out>127)
						out=127;
					//data_out.data[vv]=convert_char_rtz(out);//	Round towards zero
                    data_out.data[vv]=(char)out;//	Round towards zero
					avgPoolSum[vv]=0;
				}
				top[outnum]=data_out;
				outnum++;
			}
		}

	}
	//printf("Kernel eltwise lanched !!!\n");
}
}