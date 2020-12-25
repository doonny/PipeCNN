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
void memRead(
    // Params Ports
    uchar  data_dim1,
    uchar  data_dim2,
    ushort data_dim1xdim2,
    uchar  weight_dim1,
    uchar  weight_dim2,
    ushort weight_dim3,
    ushort weight_dim4_div_lane, // avoid generating divider
    uchar  weight_dim1x2,
    uint   weight_dim1x2x3,
    uchar  conv_x,
    //uchar  conv_y,           // not used in this version
    uchar  stride,
    uchar  padding,
    uchar  split,
    uchar  group_num_x,
    uchar  group_num_y,
    uchar  group_rem_size_x,
    //uchar  group_rem_size_y, // not used in this version
    uint   group_rem_size_xyz,
    uchar  win_size_x,
    uchar  win_size_y,
    uint   win_size_xyz,
    // Data Ports
    const lane_data    *bottom,
    const channel_vec  *weights,
    const channel_scal *bias,
	hls::stream<k2k_data_xlane>     &bias_out,
	hls::stream<k2k_data_vecxlane>  &weight_out,
	hls::stream<k2k_data_vecxlane>  &data_out
	)

{
#pragma HLS INTERFACE m_axi port = bottom  offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = weights  offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = bias  offset = slave bundle = gmem2


    // Input Data, Weights and Bias
    lane_data     data_vec;
    channel_vec   data_ch_vec;
    channel_vec   weight_ch_vec;
    channel_scal  bias_ch_in;
    ushort        data_offset = 0; // assuming the 1st layer is not in split
	k2k_data_xlane bias_out_tmp;
	k2k_data_vecxlane weight_out_tmp;
	k2k_data_vecxlane data_out_tmp;

    // virtual loop counters
    ushort gp_num_x, gp_num_y, out_idx_z;
    ushort gp_num_x_winbuf, gp_num_y_winbuf, out_idx_z_winbuf;
    uchar  output_idx_dim1=0, output_idx_dim2=0;
    ushort output_idx_dim3=0;
    uchar  win_itm_x=0, win_itm_y=0;
    ushort win_itm_z=0;

    uchar  gp_item_idx_x=0;

    ushort feature_idx_dim1, feature_idx_dim2;
    ushort feature_idx_dim3;

    uint   item_loop_bound;

    uchar  flag; // ping-pong flag

	uchar  load_weight_flag = 1;
	uchar  load_feature_flag = 1;

    // Ping-pong buffer
    lane_data    win_buffer[2][WIN_BUF_SIZE];
    #pragma HLS ARRAY_RESHAPE variable=win_buffer complete dim=1
    // Weight buffer
    channel_vec  weight_buffer[WEIGHT_BUF_SIZE];

Init:// Initialize the winbuf with the data in the first iteration of the group looping (as gp_num_x_winbuf=0, gp_num_y_winbuf=0)
    for(unsigned short win_itm_z_1=0; win_itm_z_1<weight_dim3/VEC_SIZE; win_itm_z_1++){
        for(unsigned char  win_itm_y_1=0; win_itm_y_1<win_size_y; win_itm_y_1++){
            for(unsigned char  win_itm_x_1=0; win_itm_x_1<win_size_x; win_itm_x_1++){

                feature_idx_dim1 = win_itm_x_1;
                feature_idx_dim2 = win_itm_y_1;
                feature_idx_dim3 = win_itm_z_1;

                // fetch feature map for the current group and caching in win buffer
                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2>=padding && feature_idx_dim2<data_dim2+padding)){
                    data_vec = bottom[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2-padding)*data_dim1 + (feature_idx_dim1-padding)];
                }
                else{ // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
				// or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                    for(unsigned char vv=0; vv<VEC_SIZE; vv++){
						#pragma HLS unroll
                        data_vec.data[vv] = CZERO;
                    }
                }
                // start from using buffer[0]
                win_buffer[0][win_itm_z_1*win_size_y*win_size_x + win_itm_y_1*win_size_x + win_itm_x_1] = data_vec;
            
            }
        }
    }

    // reset group virtual loop counters for winbuf loading operations
	// the gp loop counter for winbuf starts one iteration earlier than global group virtual loop counter
	// in this iteration, the winbuf is pre-initialized as previous loops shows
    if(group_num_x==1 && group_num_y==1){
        gp_num_x_winbuf = 0; // there is only one group for FC mode when batch=1
        gp_num_y_winbuf = 0;}
    else if(group_num_x==1){
		gp_num_x_winbuf = 0; // special case for convolution layers with weight_dim1/2=1, such as resnet50
		gp_num_y_winbuf = 1;}
	else{
		gp_num_x_winbuf = 1; // loop start from the second group for normal convolution layers
		gp_num_y_winbuf = 0;}

    out_idx_z_winbuf = 0;

	// reset global group virtual loop counters
	gp_num_x = 0;           // Which group in x dim
	gp_num_y = 0;           // Which group in y dim
	out_idx_z = 0;          // Which group in z dim

	// The "Group" and "Item" for loops are merged to improve pipeline efficiency.
	uint conv_x_idx = 0;    // It is used to check the validation of the output pixel, conv_x_idx = gp_num_x*CONV_GP_SIZE_X+gp_item_idx_x, and it should within [0, conv_x]
	uint out_idx_xyz = 0;   // Index of output groups, it indicates which output group.
	uint item_loop_cnt = 0; // The counter to determine the end of one group.
	uint ItemLoopBound;     // The loop tripcount within one group.
	uint ItemLastLoopBound; // The loop tripcount of the last group of each conv_out row.
	uint TotalLoopBound;    // Total loop tripcount for the entire feature map.

	if(stride>=weight_dim1 || stride>=weight_dim2) // special case convolution layers with stride>weight_dim1/2, such as resnet50
		ItemLoopBound = win_size_xyz/VEC_SIZE;
	else
		ItemLoopBound = (weight_dim1x2x3*CONV_GP_SIZE_Y*CONV_GP_SIZE_X/VEC_SIZE);
	ItemLastLoopBound = win_size_x>=group_rem_size_x?(win_size_xyz/VEC_SIZE):(group_rem_size_xyz/VEC_SIZE);

	TotalLoopBound = ((group_num_x-1)*ItemLoopBound+ItemLastLoopBound) * group_num_y * weight_dim4_div_lane;

	for(unsigned int total_cnt=0; total_cnt<TotalLoopBound; total_cnt++){
	//Group:for(unsigned int out_idx_xyz=0; out_idx_xyz<(weight_dim4_div_lane*group_num_y*group_num_x); out_idx_xyz++){
	//The following THREE loops are merged to the "Group" loop to improve pipeline efficiency,
	//    Moreover, the "Group" and the "Item" loops are further merged to the upper loop.
	//for(unsigned short out_idx_z=0; out_idx_z<weight_dim4_div_lane; out_idx_z++){//which group in dim4
	//for(unsigned short gp_num_y=0; gp_num_y<group_num_y; gp_num_y++){//which group in dim2
	//for(unsigned short gp_num_x=0; gp_num_x<group_num_x+1; gp_num_x++){ // add one more extra iteration for ping-pong buffering operations
		if(item_loop_cnt == 0){// if starting from a new group.
		    // special case when split==1, the output feature maps depend on only half the input feature maps
            if(split==0)
                data_offset = 0;
            else if(out_idx_z_winbuf<(weight_dim4_div_lane>>1)) // the lower half of the output feature maps depend on the lower half of the input
                data_offset = 0;
            else
                data_offset = weight_dim3/VEC_SIZE;	// the upper half of the output feature maps depend on the upper half of the input

                flag = out_idx_xyz & 0x01; //ping-pong flag
    			load_feature_flag = 1;

 			if(gp_num_x==group_num_x-1) // last group in each row
				// ensuring that both winbuf load loop and output loop are finished, i.e., use a larger value as the loop bound
				item_loop_bound = ItemLastLoopBound;
			else{
				item_loop_bound = ItemLoopBound;
			}
		}


		//Item:for(unsigned int  win_itm_xyz=0; win_itm_xyz<item_loop_bound; win_itm_xyz++){
		//The following THREE loops are flattened as the upper "Item" loop to improve pipeline efficiency.
		//    moreover, the "Item" loop is further merged with the "Group" loop.
		//for(unsigned short win_itm_z=0; win_itm_z<weight_dim3/VEC_SIZE; win_itm_z++){
			//for(unsigned char  win_itm_y=0; win_itm_y<weight_dim2*CONV_GP_SIZE_Y; win_itm_y++){
				//for(unsigned char  win_itm_x=0; win_itm_x<weight_dim1*CONV_GP_SIZE_X; win_itm_x++){

                            // Winbuffer loading operations
                            if(load_feature_flag==1){

                                feature_idx_dim1 = win_itm_x+gp_num_x_winbuf*CONV_GP_SIZE_X*stride;
                                feature_idx_dim2 = win_itm_y+gp_num_y_winbuf*CONV_GP_SIZE_Y*stride;
                                feature_idx_dim3 = win_itm_z;

                                // fetch feature map for the current group and caching in win buffer
                                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2>=padding && feature_idx_dim2<data_dim2+padding)){
                                    data_vec = bottom[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2-padding)*data_dim1 + (feature_idx_dim1-padding)];
                                }
                                else{ // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                                    // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                                    for(unsigned char vv=0; vv<VEC_SIZE; vv++){
										#pragma HLS unroll
                                        data_vec.data[vv] = CZERO;
                                    }
                                }

                                win_buffer[(~flag)&0x01][win_itm_z*win_size_y*win_size_x + win_itm_y*win_size_x + win_itm_x] = data_vec;
                                
						        // used as loop counters
						        if((win_itm_z==weight_dim3/VEC_SIZE-1) && (win_itm_y==win_size_y-1) && (win_itm_x==win_size_x-1)){
						        	win_itm_z = 0;
						        	load_feature_flag = 0;
						        }
						        else if((win_itm_y==win_size_y-1) && (win_itm_x==win_size_x-1)){
						        	win_itm_z++;
						        }
						        if((win_itm_y==win_size_y-1) && (win_itm_x==win_size_x-1)){
						        	win_itm_y = 0;
						        }
                                else if(win_itm_x==win_size_x-1)
                                    win_itm_y++;

                                if(win_itm_x==win_size_x-1)
                                    win_itm_x = 0;
                                else
                                    win_itm_x++;
                            }

                            // Load weight into weight buffer
                            if(load_weight_flag==1){
                                weight_ch_vec = weights[out_idx_z*weight_dim1x2x3/VEC_SIZE + output_idx_dim3*weight_dim1x2 + output_idx_dim2*weight_dim1 + output_idx_dim1];
                                weight_buffer[output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1] = weight_ch_vec;
                            }

                            // Only output data for valid convolution work-items
                            // In this version, grouping is only performed in row (x) direction
                            if(gp_num_x*CONV_GP_SIZE_X+gp_item_idx_x<conv_x){

                                if(output_idx_dim1==0 && output_idx_dim2==0 && output_idx_dim3==0){
        							if(load_weight_flag==1){
                                        bias_ch_in = bias[out_idx_z];
									    for(unsigned char ll=0; ll<LANE_NUM; ll++){
									    	#pragma HLS unroll
									    	bias_out_tmp.data(ll*DP_WIDTH+DP_WIDTH-1, ll*DP_WIDTH) = bias_ch_in.lane[ll];
									    }
                                    }
									bias_out.write(bias_out_tmp);
                                    //bias_ch_write_pipe_block(bias_ch_in);
                                    //#ifdef DEBUG_MEMRD
                                    //printf("work-item x=%d, y=%d, z=%d, channel =0, write bias=%d\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, bias_ch_in.lane[0]);
                                    //#endif
                                }

                                // data
                                data_vec = win_buffer[flag&0x01][output_idx_dim3*win_size_y*win_size_x + output_idx_dim2*win_size_x + (output_idx_dim1+gp_item_idx_x*stride)];
                                for(unsigned char ll=0; ll<LANE_NUM; ll++){
									#pragma HLS unroll
                                    //data_ch_vec.lane[ll] = data_vec;
									for(unsigned char vv=0; vv<VEC_SIZE; vv++){ // copy data_vec to each lane
										#pragma HLS unroll
										data_out_tmp.data(VEC_SIZE*DP_WIDTH*ll+(vv*DP_WIDTH+DP_WIDTH-1), VEC_SIZE*DP_WIDTH*ll+(vv*DP_WIDTH))=data_vec.data[vv];
									}
                                }
                                //data_write_pipe_block(data_ch_vec);
								data_out.write(data_out_tmp);


                                // weight and bias fetcher
                                weight_ch_vec = weight_buffer[output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1];
                                //weight_write_pipe_block(weight_ch_vec);
                                for(unsigned char ll=0; ll<LANE_NUM; ll++){
									#pragma HLS unroll
									for(unsigned char vv=0; vv<VEC_SIZE; vv++){ // copy weight to each lane
										#pragma HLS unroll
										weight_out_tmp.data(VEC_SIZE*DP_WIDTH*ll+(vv*DP_WIDTH+DP_WIDTH-1), VEC_SIZE*DP_WIDTH*ll+(vv*DP_WIDTH))=weight_ch_vec.lane[ll].data[vv];
									}
                                }
								weight_out.write(weight_out_tmp);

                                #ifdef DEBUG_MEMRD
                                if(gp_num_x==group_num_x-1 && gp_num_y==0 && out_idx_z==0){
                                    //printf("work-item x=%d, y=%d, z=%d, offset=%d, write data in channel 0=%f\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, data_offset, (float)data_ch_vec.lane[0].data[0]);
                                    printf("work-item x=%d, y=%d, z=%d, write weight in channel 0=%f\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, (float)weight_ch_vec.lane[0].data[0]);
                                }
                                #endif

                                // used as output loop counters
						        if(((gp_num_x*CONV_GP_SIZE_X+gp_item_idx_x==conv_x-1) || (gp_item_idx_x==CONV_GP_SIZE_X-1)) && (output_idx_dim3==weight_dim3/VEC_SIZE-1) && (output_idx_dim2==weight_dim2-1) && (output_idx_dim1==weight_dim1-1)){
						        	gp_item_idx_x=0;
						        }
						        else if((output_idx_dim3==weight_dim3/VEC_SIZE-1) && (output_idx_dim2==weight_dim2-1) && (output_idx_dim1==weight_dim1-1)){
						        	gp_item_idx_x++;
						        }
                                if((output_idx_dim3==weight_dim3/VEC_SIZE-1) && (output_idx_dim2==weight_dim2-1) && (output_idx_dim1==weight_dim1-1)){
                                    output_idx_dim3 = 0;
							        load_weight_flag = 0;
                                }
                                else if((output_idx_dim2==weight_dim2-1)&& (output_idx_dim1==weight_dim1-1))
                                    output_idx_dim3++;

                                if((output_idx_dim2==weight_dim2-1) && (output_idx_dim1==weight_dim1-1))
                                    output_idx_dim2 = 0;
                                else if(output_idx_dim1==weight_dim1-1)
                                    output_idx_dim2++;

                                if(output_idx_dim1==weight_dim1-1)
                                    output_idx_dim1 = 0;
                                else
                                    output_idx_dim1++;
                            }

				//}// end of win_itm_x
			//}// end of win_itm_y
		//}// end of win_itm_z

		    if(item_loop_cnt == item_loop_bound-1){//one group is finished
			    item_loop_cnt = 0;
			    out_idx_xyz++;	
                // used as virtual group loop counters for winbuf loading operations
                if((out_idx_z_winbuf==weight_dim4_div_lane-1) && (gp_num_y_winbuf==group_num_y-1) && (gp_num_x_winbuf==group_num_x-1))
                    out_idx_z_winbuf = 0;
                else if((gp_num_y_winbuf==group_num_y-1) && (gp_num_x_winbuf==group_num_x-1))
                    out_idx_z_winbuf++;

                if((gp_num_y_winbuf==group_num_y-1) && (gp_num_x_winbuf==group_num_x-1))
                    gp_num_y_winbuf = 0;
                else if(gp_num_x_winbuf==group_num_x-1)
                    gp_num_y_winbuf++;

                if(gp_num_x_winbuf==group_num_x-1)
                    gp_num_x_winbuf = 0;
                else
                    gp_num_x_winbuf++;

			    // used as virtual group loop counters for sending data to the conv kernel
                if((out_idx_z==weight_dim4_div_lane-1) && (gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
                    out_idx_z = 0;
			    else if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1)){
                    out_idx_z++;
                 	load_weight_flag = 1;
                }

                if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
                    gp_num_y = 0;
                else if(gp_num_x==group_num_x-1)
                    gp_num_y++;

                if(gp_num_x==group_num_x-1)
                    gp_num_x = 0;
                else
                    gp_num_x++;
            }
		    else{//one group is not finished, continued.
		    	item_loop_cnt++;
		    }
	//			}// end of gp_num_x
	//		}// end of gp_num_y
	//}// end of out_idx_z
    }

    //printf("Kernel 0 lanched !!!\n");
}
}
