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


DPTYPE pool_max(DPTYPE a_in, DPTYPE b_in)
{
#pragma HLS inline

    DPTYPE max_value;

    if(a_in >= b_in)
        max_value = a_in;
    else
        max_value = b_in;

    return max_value;

}

extern "C" {
void maxPool(
		// Params Ports
		uchar    conv_x,
		ushort   conv_xy,
		uchar    pool_dim1,
		ushort   pool_dim3,
		ushort   pool_dim1x2,
		uchar    pool_size,
		uchar    pool_stride,
		uchar    padd_offset,
		ushort   pool_times, //pool_group*pool_y
		ushort   pool_group, //the number of pooling z dimension vectorized packet
		ushort   pool_y_bound, //pooling bound per pool_y(item_loop_bound*(pool_win_num_x+1))
		ushort   item_loop_bound, // maximum of load_data_bound and write_back_bound
		ushort   load_data_bound, // pooling window buffer load data bound
		ushort   write_back_bound,// pooling window buffer write back result to global memory bound
		uchar    pool_win_num_x, //the number of pool window buffer
		uchar    win_size_x, // pool window buffer size of x dimension
		const channel_scal    *bottom,
		DPTYPE                *top
		//hls::stream<k2k_sync> &pool_sync_in
		)
{
#pragma HLS INTERFACE m_axi port = bottom  offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = top  offset = slave bundle = gmem1

    uchar pool_sync=0; // Receive channel synchronization signal
	uint  base_addr; // basic address of global memory
	uchar flag; // ping-pong buffer flag
	k2k_sync pool_sync_tmp;

	// the counter of pooling hierarchy
	ushort  pool_group_cnt; // pooling z dimension vectorized packet counter
	uchar   pool_y_cnt; // pooling y dimension counter
	ushort  item_loop_cnt; // the counter of item_loop_bound
	ushort  pool_win_cnt; // the counter of pool_win_num_x(ping-pong +1)
	// the counter of pool window buffer
	uchar   win_item_x; // x dimension
	uchar   win_item_y; // y dimension
	uchar   win_item_s; // pool stride in pool window buffer
	uchar   win_item_cnt; // for win_item_s
	// the counter of write back
	uchar   gp_final_cnt; // pooling result counter in window buffer
	uchar   lane_cnt;
	ushort  global_z;
	ushort  global_index_z_group;
	uchar   global_index_z_item;
	// the register of pooling
	DPTYPE  shift_reg[LANE_NUM][POOL_MAX_SIZE]; // cache from global memory
	DPTYPE  temp_reg0[LANE_NUM];
	DPTYPE  temp_reg1[LANE_NUM];
	DPTYPE  temp_reg2[LANE_NUM];
	DPTYPE  temp_reg[LANE_NUM];
	DPTYPE  temp_max[LANE_NUM];

	DPTYPE  row_reg0[LANE_NUM][POOL_GP_SIZE_X]; // pooling reslut in the first line
	DPTYPE  row_reg1[LANE_NUM][POOL_GP_SIZE_X]; // pooling reslut in the max(second line , first line)
	DPTYPE  pool_final[2][POOL_GP_SIZE_X][LANE_NUM]; // final pooling reslut

	// init hierarchy counters
	pool_y_cnt = 0;
	pool_group_cnt = 0;
	//#pragma ivdep array(pool_final)
	for(ushort i = 0; i < pool_times; i++){
		//pool_sync_ch_read_pipe_block(pool_sync);
		//pool_sync_tmp = pool_sync_in.read();
		//pool_sync = pool_sync_tmp.data(7,0);
		
		//if(pool_sync == 1){// sync signal
		//printf("Received Sync i=%d!!!\n", i);

		// init counters
		pool_win_cnt = 0;
		item_loop_cnt = 0;
		win_item_x = 0;
		win_item_y = 0;
		win_item_s = 0;
		win_item_cnt = 0;
		gp_final_cnt = 0;
		lane_cnt = 0;

		//#pragma ivdep array(pool_final)
		for(ushort k=0; k<pool_y_bound; k++){
			flag = pool_win_cnt & 0x01;
			base_addr = pool_group_cnt*conv_xy + pool_stride*conv_x*pool_y_cnt + pool_stride*pool_win_cnt*POOL_GP_SIZE_X;

			// load data from global memory to shift registers and pool (0--pool_win_num_x-1)
			if((pool_win_cnt < pool_win_num_x)&&(item_loop_cnt < load_data_bound)){

				if(win_item_x > pool_size-1){
					win_item_cnt++;
				}
				//__attribute__((opencl_unroll_hint))
				for(uchar ll=0; ll<LANE_NUM; ll++){
					#pragma HLS unroll

					if( (pool_win_cnt*POOL_GP_SIZE_X*pool_stride+win_item_x) < conv_x){
						// load global memory to shift register
						shift_reg[ll][0] = bottom[base_addr+win_item_y*conv_x+win_item_x].lane[ll];

						// fetch the data from shift register to pool
						if((win_item_x == pool_size-1) || (win_item_cnt == pool_stride)){
							temp_reg0[ll] = shift_reg[ll][0];
							temp_reg1[ll] = shift_reg[ll][1];
							temp_reg2[ll] = shift_reg[ll][2];
						}

						else{
							temp_reg0[ll] = CZERO;
							temp_reg1[ll] = CZERO;
							temp_reg2[ll] = CZERO;
						}
						// pooling for pool size equal 3
						if(pool_size == 3){
							temp_reg[ll] = pool_max(temp_reg0[ll],temp_reg1[ll]);
							temp_max[ll] = pool_max(temp_reg2[ll],temp_reg[ll]);
							if(win_item_y == 0){
								row_reg0[ll][win_item_s] = temp_max[ll];
							}
							else if(win_item_y == 1){
								row_reg1[ll][win_item_s] = pool_max(temp_max[ll],row_reg0[ll][win_item_s]);
							}
							else{
								pool_final[flag][win_item_s][ll] = pool_max(temp_max[ll],row_reg1[ll][win_item_s]);
							}
						}
						// pooling for pool size equal 2
						else{
							temp_max[ll] = pool_max(temp_reg1[ll],temp_reg0[ll]);
							if(win_item_y == 0){
								row_reg0[ll][win_item_s] = temp_max[ll];
							}
							else{
								pool_final[flag][win_item_s][ll] = pool_max(temp_max[ll],row_reg0[ll][win_item_s]);
							}
						}

						// shift register
						//__attribute__((opencl_unroll_hint))
						for(uchar p=POOL_MAX_SIZE-1; p>0; p--){
							#pragma HLS unroll

							shift_reg[ll][p] = shift_reg[ll][p-1];
						}
					}
				}


				if((win_item_x == pool_size-1)||(win_item_cnt == pool_stride)){
					win_item_s++;
				}

				if(win_item_cnt == pool_stride){
					win_item_cnt = 0;
				}

				if((win_item_y == pool_size-1) && (win_item_x == win_size_x-1))
					win_item_y = 0;
				else if(win_item_x == win_size_x-1)
					win_item_y++;
				if(win_item_x == win_size_x-1)
					win_item_x = 0;
				else
					win_item_x++;

				if(win_item_x == 0)
					win_item_s = 0;

			}

			// write back result to global memoey
			// perform vectorization in dim3 (global_z) by combining multiple DPTYPE data into lane_data type
			if((pool_win_cnt > 0) && (item_loop_cnt < write_back_bound)){
				if(((pool_win_cnt-1)*POOL_GP_SIZE_X+gp_final_cnt) < pool_dim1){
					global_z = pool_group_cnt*LANE_NUM+lane_cnt;
					global_index_z_group = (global_z-padd_offset) / VEC_SIZE;
					global_index_z_item =  (global_z-padd_offset) % VEC_SIZE;
					if((global_z-padd_offset)<pool_dim3 && global_z>=padd_offset){
						top[global_index_z_group*pool_dim1x2*VEC_SIZE+pool_y_cnt*pool_dim1*VEC_SIZE+((pool_win_cnt-1)*POOL_GP_SIZE_X+gp_final_cnt)*VEC_SIZE+global_index_z_item] = pool_final[!flag][gp_final_cnt][lane_cnt];
					}
				}

				if((gp_final_cnt == POOL_GP_SIZE_X-1) && (lane_cnt == LANE_NUM-1))
					gp_final_cnt = 0;
				else if(lane_cnt == LANE_NUM-1)
					gp_final_cnt++;
				if(lane_cnt == LANE_NUM-1)
					lane_cnt = 0;
				else
					lane_cnt++;
			}

			if((pool_win_cnt == pool_win_num_x) && (item_loop_cnt == item_loop_bound-1))
				pool_win_cnt = 0;
			else if(item_loop_cnt == item_loop_bound-1)
				pool_win_cnt++;
			if(item_loop_cnt == item_loop_bound-1)
				item_loop_cnt = 0;
			else
				item_loop_cnt++;
		}

		//}// end of sync

		if((pool_group_cnt == pool_group-1) && (pool_y_cnt == pool_dim1-1))
			pool_group_cnt = 0;
		else if(pool_y_cnt == pool_dim1-1)
			pool_group_cnt++;
		if(pool_y_cnt == pool_dim1-1)
			pool_y_cnt = 0;
		else
			pool_y_cnt++;
	}
}
}