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


// Store Data to Global Memory
extern "C" {
void memWrite(
				// Params Ports
				uchar  out_dim1,
				uchar  out_dim2,
				ushort out_dim3,
				ushort out_dim1xbatch, // out_dim1 x sqrt(batch_size)
				uint   out_dim1x2xbatch, // out_dim1 x out_dim2 x batch_size
				uchar  batch_indx_dim1,
				uchar  batch_indx_dim2,
#ifdef RESNET
				uchar  bypass,
				uchar  pool_pad,	  //RESNET need pad,set to 1,other CNN set 0
#endif
				uchar  padd_offset,
				uchar  pool_on,
				uchar  pool_size,
				uchar  pool_stride,
				// Data Ports
				DPTYPE *top,
				// Stream Ports
#ifdef RESNET
				hls::stream<k2k_data_xlane>     &bn_in,
				hls::stream<k2k_data_xlane>     &bypass_in
#else
				hls::stream<k2k_data_xlane>     &conv_in
				//hls::stream<k2k_sync>           &pool_sync_out
#endif
)
{
#pragma HLS INTERFACE m_axi port = top  offset = slave bundle = gmem0

    uchar index_z_item; // max value 256
    ushort index_z_group; // max value 4096
	uint top_addr;
	uchar pool_on_signal=1;

	k2k_data_xlane conv_in_tmp;
	k2k_sync       pool_sync_tmp;

    channel_scal output;
    DPTYPE buffer[LANE_NUM];
	#pragma HLS ARRAY_PARTITION variable=buffer dim=1 complete

	ushort out_idx_x = 0;
	ushort out_idx_y = 0;
	uchar  lane_item_idx = 0; //index within one LANE, between [0, LANE_NUM-1]
	ushort lane_num_idx = 0;  //index of which LANE, between [0, conv_out_dim3/LANE_NUM-1], NOTE: conv_out_dim3 includes the LANE padding.
	
	for(unsigned int i=0; i<(out_dim1*out_dim2*(out_dim3+2*padd_offset)); i++) {
	//The following loops are merged
    //for(unsigned int loop_z=0; loop_z<out_dim3/LANE_NUM; loop_z++) {
    //    for(unsigned int loop_y=0; loop_y<out_dim2; loop_y++) {
    //        for(unsigned int loop_x=0; loop_x<out_dim1; loop_x++) {
    //            for(unsigned int loop=0; loop<LANE_NUM; loop++) {
					//The begining of receiving one lane data
                    if(lane_item_idx==0) {
#ifdef RESNET
							if(bypass==0){//bypass==0,bn
								// for ResNet padding
								if((pool_on == 1) && ((out_idx_y >= out_dim2-pool_pad) || ((out_idx_x >= out_dim1-pool_pad)))){
										for(uchar ll=0; ll<LANE_NUM; ll++){
											#pragma HLS unroll
											output.lane[ll]=-128;
										}
								}
								else{
									conv_in_tmp = bn_in.read();
									for(unsigned char ll=0; ll<LANE_NUM; ll++){
										#pragma HLS unroll
										output.lane[ll] = conv_in_tmp.data(ll*DP_WIDTH+DP_WIDTH-1, ll*DP_WIDTH);
									}
									//batchNorm_ch_read_pipe_block(output);
								}
							}
							else{ // bypass == 1 bypass_bn_ch
								conv_in_tmp = bypass_in.read();
								for(unsigned char ll=0; ll<LANE_NUM; ll++){
									#pragma HLS unroll
									output.lane[ll] = conv_in_tmp.data(ll*DP_WIDTH+DP_WIDTH-1, ll*DP_WIDTH);
								}
								//bypass_bn_ch_read_pipe_block(output);
							}
#else
							conv_in_tmp = conv_in.read();
							for(unsigned char ll=0; ll<LANE_NUM; ll++){
								#pragma HLS unroll
								output.lane[ll] = conv_in_tmp.data(ll*DP_WIDTH+DP_WIDTH-1, ll*DP_WIDTH);
							}
							//conv_ch_read_pipe_block(output);
#endif
						
						// store the vectorized output into local buffer
						for(uchar ll=0; ll<LANE_NUM; ll++){
							#pragma HLS unroll
							buffer[ll]=output.lane[ll];
						}
                	}

					// fetch data from local buffer and write back to DDR
					// perform vectorization in dim3 (global_z) by combining multiple DPTYPE data into lane_data type
					if(pool_on == 1){
						top_addr = lane_num_idx*out_dim1x2xbatch*LANE_NUM +(out_idx_y+batch_indx_dim2*out_dim2)*out_dim1xbatch*LANE_NUM + (out_idx_x+batch_indx_dim1*out_dim1)*LANE_NUM + lane_item_idx;
					}
					else{
						index_z_group = (lane_num_idx*LANE_NUM+lane_item_idx-padd_offset)/VEC_SIZE;
						index_z_item  = (lane_num_idx*LANE_NUM+lane_item_idx-padd_offset)%VEC_SIZE;
						top_addr = index_z_group*out_dim1x2xbatch*VEC_SIZE + (out_idx_y+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (out_idx_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
					}

					// output dim3 in current layer may be larger than next layer (the value is changed to a value of multiples of LANE_NUM to saturated the wide pipeline input)
					// therefore, only write back the valid values without padding zeros
					if((lane_num_idx*LANE_NUM+lane_item_idx-padd_offset)<out_dim3 && (lane_num_idx*LANE_NUM+lane_item_idx>=padd_offset)){
						// 1. addressing expression with out batch processing is
						// top[index_z_group*dim1*dim2*VEC_SIZE + global_y*dim1*VEC_SIZE + global_x*VEC_SIZE + index_z_item]=buffer[local_z];
						// 2. addressing expression with batch processing (batch_size_in_dim = sqrt(batch_size)) is
						// top[(index_z_group*out_dim2*out_dim1*batch_size_in_dim*batch_size_in_dim*VEC_SIZE + (global_y+batch_indx_dim2*out_dim2)*batch_size_in_dim*out_dim1*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item] = buffer[local_z];
						// 3. simplified addressing with reduced cost of multipliers
						//printf("b=%d\n",index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item);

						top[top_addr] = buffer[lane_item_idx];
						#ifdef DEBUG_MEMWR
						//if((global_z-padd_offset) == 0){
							//for(unsigned char ll=0; ll<LANE_NUM; ll++){
							printf("MemWr results= %f (x=%d, y=%d, z=%d, ll=%d)\n", (float)output.lane[0], out_idx_x, out_idx_y, lane_num_idx*LANE_NUM+lane_item_idx, 0);
							//}
						//	}
						#endif

					}

					//if(pool_on == 1){
					//	if((loop_x==out_dim1-1)&&(loop_y > 0)&&((loop_y-pool_size+1)%2 == 0)&&(loop ==LANE_NUM-1)){
					//		//pool_sync_ch_write_pipe_block(pool_on_signal);
					//		pool_sync_tmp.data(7,0) = pool_on_signal;
					//		pool_sync_out.write(pool_sync_tmp);
					//	}
					//}

        			if((lane_num_idx==(out_dim3+2*padd_offset)/LANE_NUM-1) && (out_idx_y==out_dim2-1) && (out_idx_x==out_dim1-1) && (lane_item_idx==LANE_NUM-1))
        			    lane_num_idx = 0;
        			else if((out_idx_y==out_dim2-1) && (out_idx_x==out_dim1-1) && (lane_item_idx==LANE_NUM-1))
        			    lane_num_idx++;

        			if((out_idx_y==out_dim2-1) && (out_idx_x==out_dim1-1) && (lane_item_idx==LANE_NUM-1))
        			    out_idx_y = 0;
        			else if((out_idx_x==out_dim1-1) && (lane_item_idx==LANE_NUM-1))
        			    out_idx_y++;
					
        			if((out_idx_x==out_dim1-1) && (lane_item_idx==LANE_NUM-1))
        			    out_idx_x = 0;
        			else if(lane_item_idx==LANE_NUM-1)
        			    out_idx_x++;

        			if(lane_item_idx==LANE_NUM-1)
        			    lane_item_idx = 0;
        			else
        			    lane_item_idx++;
    //            }
    //        }
    //    }
    //}
	}// end of merged loop
}
}
