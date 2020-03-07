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
 *   Copyright (C) 2016, Institute of Information Science,
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

#include "hw_param.cl"
#include "rtl_lib.h"

#pragma OPENCL EXTENSION cl_intel_channels : enable
#pragma OPENCL EXTENSION cl_intel_arbitrary_precision_integers : enable

// Define the precision of the data-path
typedef char DPTYPE;
typedef int  MACTYPE;

// Vectorized data type
typedef struct {
    DPTYPE data[VEC_SIZE];
} DPTYPE_VEC;

// Vectorized data type
typedef struct {
    DPTYPE_VEC data[PE_NUM_Y];
} DPTYPE_PE_VEC;

// Combined vec-data type from multiple lane
typedef struct {
    DPTYPE_PE_VEC lane[LANE_NUM];
} SCAL_PE_VEC;

typedef struct {
    DPTYPE_VEC lane[LANE_NUM];
} DPTYPE_SCAL_VEC;

typedef struct {
    DPTYPE_SCAL_VEC data[PE_NUM_Y];
} PE_SCAL_VEC;

// Combined scalar data type from multiple lane
typedef struct {
    DPTYPE lane[LANE_NUM];
} DPTYPE_SCAL;

typedef struct {
    DPTYPE_SCAL data[PE_NUM_Y];
} DPTYPE_PE_SCAL;

#if defined BN_FLOAT || defined RELU_FP
typedef struct {
    float lane[LANE_NUM];
} FLOAT_SCAL;

typedef struct {
    FLOAT_SCAL data[PE_NUM_Y];
} FLOAT_PE_scal;
#endif

#ifdef CHANNEL_OPT
channel DPTYPE_PE_VEC    data_ch    __attribute__((depth(0)));
channel DPTYPE_SCAL_VEC    weight_ch  __attribute__((depth(0)));
#else
channel SCAL_PE_VEC    data_ch    __attribute__((depth(0)));
channel PE_SCAL_VEC    weight_ch  __attribute__((depth(0)));
#endif
channel DPTYPE_SCAL   bias_ch    __attribute__((depth(8)));
channel DPTYPE_PE_SCAL   conv_ch    __attribute__((depth(CHN_DEPTH)));
channel bool           pool_sync_ch __attribute__((depth(8)));
#if defined BN_FLOAT || defined RELU_FP
channel DPTYPE_PE_SCAL   	batchNorm_ch  __attribute__((depth(CHN_DEPTH)));
channel DPTYPE_PE_SCAL   	bypass_bn_ch  __attribute__((depth(CHN_DEPTH)));
#endif


#ifdef ARBI_PRECISION
// parallel MAC units including (VEC_SIZE-1) multipliers
ap_int<SUM_BIT> mac(DPTYPE_VEC input, DPTYPE_VEC weights) {
    ap_int<SUM_BIT> output = MASK_MULT & CZERO;

#pragma unroll
    for(int i=0; i<VEC_SIZE/4; i++) {
        //output += input.data[i]*weights.data[i];
        // use packed DSP blocks to improve efficiency
        output += MASK_MULT & mult_add_fix8bx4(input.data[i*4], weights.data[i*4], input.data[i*4+1], weights.data[i*4+1], input.data[i*4+2], weights.data[i*4+2], input.data[i*4+3], weights.data[i*4+3]);
    }
    return output;
}
#else
// parallel MAC units including (VEC_SIZE-1) multipliers
MACTYPE mac(DPTYPE_VEC input, DPTYPE_VEC weights) {
    MACTYPE output = MASK_MULT & CZERO;

#pragma unroll
    for(int i=0; i<VEC_SIZE/4; i++) {
        //output += input.data[i]*weights.data[i];
        // use packed DSP blocks to improve efficiency
        output += MASK_MULT & mult_add_fix8bx4(input.data[i*4], weights.data[i*4], input.data[i*4+1], weights.data[i*4+1], input.data[i*4+2], weights.data[i*4+2], input.data[i*4+3], weights.data[i*4+3]);
    }
    return output;
}
#endif

DPTYPE pool_max(DPTYPE a_in, DPTYPE b_in) {
    DPTYPE max_value;

    if(a_in >= b_in)
        max_value = a_in;
    else
        max_value = b_in;

    return max_value;
}


#if defined RD_PORT
// Fetch Data from Global Memory
__kernel
__attribute__((task))
__attribute__((max_global_work_dim(0)))
void memRead(
    // Params Ports
    ushort  data_dim1,
    ushort  data_dim2,
    uint data_dim1xdim2,
    uchar  weight_dim1,
    uchar  weight_dim2,
    ushort weight_dim3,
    ushort weight_dim4_div_lane, // avoid generating divider
    uchar  weight_dim1x2,
    uint   weight_dim1x2x3,
    ushort  conv_x,
    //uchar  conv_y,           // not used in this version
    uchar  stride,
    uchar  padding,
    uchar  split,
    uchar gp_size_y,
    ushort gp_last_size_x,
    ushort gp_size_x,
    ushort  group_num_x,
    ushort  group_num_y,
    uint  conv_group_num_dim1x2_dim4_div_lane,
    // uchar  group_rem_size_x,
    //uchar  group_rem_size_y, // not used in this version
    // uint   group_rem_size_xyz,
    ushort win_size_x,
    // uchar  win_size_y_port,
    uint   win_size_xyz_port,
    uint Item_Loop_Bound,
    uint Item_Last_Loop_Bound,
    // Data Ports
    __global DPTYPE_VEC    *restrict bottom0,
    // __global DPTYPE_VEC    *restrict bottom1,
    // __global DPTYPE_VEC    *restrict bottom2,
    // __global DPTYPE_VEC    *restrict bottom3,
    // __global DPTYPE_VEC    *restrict bottom4,
    __global DPTYPE_SCAL_VEC  *restrict weights,
    __global volatile DPTYPE_SCAL *restrict bias        
){

    // feature win buffer
    // Ping-pong buffer
    __local DPTYPE_VEC  feature_win_buffer_0[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_1[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_2[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_3[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_4[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_5[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_6[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_7[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_8[2][WIN_BUF_DEPTH];
    // for padding when weight_w is 3
    __local DPTYPE_VEC  feature_win_buffer_9[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_10[2][WIN_BUF_DEPTH];
#if PE_NUM_Y>9
    __local DPTYPE_VEC  feature_win_buffer_11[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_12[2][WIN_BUF_DEPTH];
#endif
#if PE_NUM_Y>11
    __local DPTYPE_VEC  feature_win_buffer_13[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_14[2][WIN_BUF_DEPTH];
#endif
#if PE_NUM_Y>13
    __local DPTYPE_VEC  feature_win_buffer_15[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_16[2][WIN_BUF_DEPTH];
#endif
#if PE_NUM_Y>15
    __local DPTYPE_VEC  feature_win_buffer_17[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_18[2][WIN_BUF_DEPTH];
#endif

    // Weight buffer
#ifdef DEF_BUF_WIDTH
    __local DPTYPE_VEC DEFINE_BUF(weight_buffer, WEIGHT_BUF_SIZE);
#else
    __local DPTYPE_SCAL_VEC  weight_buffer[WEIGHT_BUF_SIZE];
#endif

    // Input Data, Weights and Bias
    // read from feature_win_buffer_0 (1,2,3,...,14)
    DPTYPE_VEC    feature_y_parallel[PE_NUM_Y+WIN_BUF_Y_PAD];
    // take data of PE_NUM_Y from feature_y_parallel
    // DPTYPE_PE_VEC    feature_y;
    // read from bottom
    DPTYPE_VEC    data_vec_port0;
    DPTYPE_VEC    data_vec_port1;
    DPTYPE_VEC    data_vec_port2;
    DPTYPE_VEC    data_vec_port3;
    DPTYPE_VEC    data_vec_port4;
    // write to kl_conv
    SCAL_PE_VEC   data_ch_vec;
    DPTYPE_SCAL_VEC weight_ch_tmp;
    PE_SCAL_VEC   weight_ch_vec;
    DPTYPE_SCAL  bias_ch_in;

    // feature win buffer read/write address
    // ushort     feature_win_buffer_rd_addr, feature_win_buffer_wr_addr;
    ushort     data_offset = 0; // assuming the 1st layer is not in split
    uchar  flag; // ping-pong flag

    // virtual loop counters
    // for loading feature to feature_win_buffer from bottom
    ushort win_itm_z;

	uchar  win_itm_y_port;
	// uchar  win_itm_y_cycling, win_itm_y_offset;
    // for transfer data to knl_conv from feature_win_buffer
    ushort output_idx_dim3;
    uchar  output_idx_dim1, output_idx_dim2;
    // uchar  idx_y_cycling, idx_y_offset;
    // for group counters
    short gp_num_x;
    ushort gp_num_y, out_idx_z;
    ushort gp_num_x_winbuf, gp_num_y_winbuf, out_idx_z_winbuf;

    // for global index
    ushort feature_idx_dim1;  // for yolov2, maximum is 418(padding)
    ushort feature_idx_dim2_port0; // for yolov2, maximum is 418(padding)
    ushort feature_idx_dim2_port1;
    ushort feature_idx_dim2_port2;
    ushort feature_idx_dim2_port3;
    ushort feature_idx_dim2_port4;
    ushort feature_idx_dim3;
    // counter the number of x-dimensional convolutions in the feature_win_buffer
    ushort  gp_item_idx_x;
    ushort  win_itm_x;
    ushort  win_item;
    uint   item_loop_bound;


    gp_num_x_winbuf = 0; // there is only one group for FC mode when batch=1
    gp_num_y_winbuf = 0;
    out_idx_z_winbuf = 0;

    // reset global group virtual loop counters
    gp_num_x = -1;
    gp_num_y = 0;
    out_idx_z = 0;


#ifdef DEF_BUF_WIDTH
#pragma ivdep array(weight_buffer_0)
#pragma ivdep array(weight_buffer_1)
#pragma ivdep array(weight_buffer_2)
#else
#pragma ivdep array(weight_buffer)
#endif                                    
Group:
    //printf("%d\n", weight_dim4_div_lane*group_num_y*group_num_x);
    //printf("%d\t%d\t%d\n", weight_dim4_div_lane, group_num_y, group_num_x);
    // printf("memRd: %d\n", conv_group_num_dim1x2_dim4_div_lane);

    for(unsigned int out_idx_xyz=0; out_idx_xyz<=conv_group_num_dim1x2_dim4_div_lane; out_idx_xyz++) {
        // The following group loops are flattened as the upper loop to improve pipeline efficiency
        //for(unsigned short out_idx_z=0; out_idx_z<weight_dim4_div_lane; out_idx_z++){
        
        // special case when split==1, the output feature maps depend on only half the input feature maps
        if(split==0)
            data_offset = 0;
        else if(out_idx_z_winbuf<(weight_dim4_div_lane>>1)) // the lower half of the output feature maps depend on the lower half of the input
            data_offset = 0;
        else
            data_offset = weight_dim3/VEC_SIZE;	// the upper half of the output feature maps depend on the upper half of the input

        //for(unsigned short gp_num_y=0; gp_num_y<group_num_y; gp_num_y++){
        //for(unsigned short gp_num_x=0; gp_num_x<group_num_x+1; gp_num_x++){ // add one more extra iteration for ping-pong buffering operations

        flag = out_idx_xyz & 0x01; //ping-pong flag

        // feature win buffer read/write address
        ushort     feature_win_buffer_rd_addr, feature_win_buffer_wr_addr;
#ifdef DEF_BUF_WIDTH
        ushort     weight_addr;
#endif
        uchar  win_itm_y_cycling, win_itm_y_offset;
        uchar  idx_y_cycling, idx_y_offset;
        // PE_SCAL_VEC    feature_y;
        DPTYPE_PE_VEC feature_y;                                        
        // reset output loop counters
        output_idx_dim1 = 0;
        output_idx_dim2 = 0;
        output_idx_dim3 = 0;
        // reset in-group item counters
        gp_item_idx_x = 0;
        
        win_itm_y_offset = 0;
        // reset input winbuffer loop counters
        win_itm_x = 0;
        win_itm_y_port = 0;
        win_itm_z = 0;


        if(gp_num_x==group_num_x-1) {// last group in each row
            // ensuring that both winbuf load loop and output loop are finished, i.e., use a larger value as the loop bound
            // item_loop_bound = win_size_x>=group_rem_size_x?(win_size_xyz/VEC_SIZE):(group_rem_size_xyz/VEC_SIZE);
            item_loop_bound = Item_Last_Loop_Bound;
            win_item = gp_last_size_x;
        }else {
            // if(stride>=weight_dim1 || stride>=weight_dim2) // special case convolution layers with stride>weight_dim1/2, such as resnet50
            //     item_loop_bound = win_size_xyz/VEC_SIZE;
            // else
                // item_loop_bound = (weight_dim1x2x3*CONV_GP_SIZE_Y*CONV_GP_SIZE_X/VEC_SIZE);
                item_loop_bound = Item_Loop_Bound;
                win_item = gp_size_x;
        }

#pragma ivdep array(feature_win_buffer_0)
#pragma ivdep array(feature_win_buffer_1)
#pragma ivdep array(feature_win_buffer_2)
#pragma ivdep array(feature_win_buffer_3)
#pragma ivdep array(feature_win_buffer_4)
#pragma ivdep array(feature_win_buffer_5)
#pragma ivdep array(feature_win_buffer_6)
#pragma ivdep array(feature_win_buffer_7)
#pragma ivdep array(feature_win_buffer_8)
#pragma ivdep array(feature_win_buffer_9)
#pragma ivdep array(feature_win_buffer_10)
#if PE_NUM_Y>9
#pragma ivdep array(feature_win_buffer_11)
#pragma ivdep array(feature_win_buffer_12)
#endif
#if PE_NUM_Y>11
#pragma ivdep array(feature_win_buffer_13)
#pragma ivdep array(feature_win_buffer_14)
#endif
#if PE_NUM_Y>13
#pragma ivdep array(feature_win_buffer_15)
#pragma ivdep array(feature_win_buffer_16)
#endif
#if PE_NUM_Y>15
#pragma ivdep array(feature_win_buffer_17)
#pragma ivdep array(feature_win_buffer_18)
#endif
#ifdef DEF_BUF_WIDTH
#pragma ivdep array(weight_buffer_0)
#pragma ivdep array(weight_buffer_1)
#pragma ivdep array(weight_buffer_2)
#else
#pragma ivdep array(weight_buffer)
#endif
Item:
        for(unsigned int  win_itm_xyz=0; win_itm_xyz<item_loop_bound; win_itm_xyz++) {
            //// The following loops are flattened as the upper loop to improve pipeline efficiency
            //for(unsigned short win_itm_z=0; win_itm_z<weight_dim3/VEC_SIZE; win_itm_z++){
            //	for(unsigned char  win_itm_y=0; win_itm_y<weight_dim2*CONV_GP_SIZE_Y; win_itm_y++){
            //		for(unsigned char  win_itm_x=0; win_itm_x<weight_dim1*CONV_GP_SIZE_X; win_itm_x++){

            // Winbuffer loading operations
            if(win_itm_z<(weight_dim3/VEC_SIZE) && out_idx_xyz<conv_group_num_dim1x2_dim4_div_lane) {
            
#ifdef STRIDE_1
		        if(stride==1){
		        	feature_win_buffer_rd_addr = win_itm_z*win_size_x + win_itm_x;
		        }
#endif
#ifdef STRIDE_4
		        else if(stride==4){
		        	feature_win_buffer_rd_addr = win_itm_z*win_size_x*4 + win_itm_y_cycling*win_size_x + win_itm_x;
		        }
#endif

                feature_idx_dim1 = win_itm_x + gp_num_x_winbuf*gp_size_x*stride;
                feature_idx_dim2_port0 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride;
                // feature_idx_dim2_port1 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride + PE_NUM_Y_DIV;
                // feature_idx_dim2_port2 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride + PE_NUM_Y_DIV*2;
                // feature_idx_dim2_port3 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride + PE_NUM_Y_DIV*3;
                // feature_idx_dim2_port4 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride + PE_NUM_Y_DIV*4;
                feature_idx_dim3 = win_itm_z;

                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port0>=padding && feature_idx_dim2_port0<data_dim2+padding)) {
                    data_vec_port0 = bottom0[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port0-padding)*data_dim1 + (feature_idx_dim1-padding)];
                } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                            // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                    #pragma unroll
                    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                        data_vec_port0.data[vv] = CZERO;
                    }
                }
                switch(win_itm_y_offset)
                {
                case 0:
                    feature_win_buffer_0[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 1:
                    feature_win_buffer_1[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 2:
                    feature_win_buffer_2[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 3:
                    feature_win_buffer_3[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 4:
                    feature_win_buffer_4[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 5:
                    feature_win_buffer_5[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 6:
                    feature_win_buffer_6[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 7:
                    feature_win_buffer_7[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 8:
                    feature_win_buffer_8[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 9:
                    feature_win_buffer_9[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 10:
                    feature_win_buffer_10[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
#if PE_NUM_Y>9
                case 11:
                    feature_win_buffer_11[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 12:
                    feature_win_buffer_12[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
#endif
#if PE_NUM_Y>11
                case 13:
                    feature_win_buffer_13[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 14:
                    feature_win_buffer_14[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
#endif
#if PE_NUM_Y>13
                case 15:
                    feature_win_buffer_15[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 16:
                    feature_win_buffer_16[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
#endif
#if PE_NUM_Y>15
                case 17:
                    feature_win_buffer_17[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 18:
                    feature_win_buffer_18[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
#endif
                default:
                    break;
                }

                // if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port1>=padding && feature_idx_dim2_port1<data_dim2+padding)) {
                //     data_vec_port1 = bottom1[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port1-padding)*data_dim1 + (feature_idx_dim1-padding)];
                // } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                //             // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                //     #pragma unroll
                //     for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                //         data_vec_port1.data[vv] = CZERO;
                //     }
                // }
                // switch(win_itm_y_offset)
                // {
                // case 0:
                //     feature_win_buffer_3[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                // case 1:
                //     feature_win_buffer_4[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                // case 2:
                //     feature_win_buffer_5[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                // // case 3:
                // //     feature_win_buffer_8[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                // // case 4:
                // //     feature_win_buffer_9[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                // }

                // if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port2>=padding && feature_idx_dim2_port2<data_dim2+padding)) {
                //     data_vec_port2 = bottom2[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port2-padding)*data_dim1 + (feature_idx_dim1-padding)];
                // } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                //             // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                //     #pragma unroll
                //     for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                //         data_vec_port2.data[vv] = CZERO;
                //     }
                // }
                // // if((win_item_y_port+PE_NUM_Y_DIV*2)<gp_size_y) //for 1x1 conv, don't need padding
                // switch(win_itm_y_offset)
                // {
                // case 0:
                //     feature_win_buffer_6[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                // case 1:
                //     feature_win_buffer_7[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                // case 2:
                //     feature_win_buffer_8[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                // // case 3:
                // //     feature_win_buffer_13[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                // // case 4:
                // //     feature_win_buffer_14[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                // }

                // if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port3>=padding && feature_idx_dim2_port3<data_dim2+padding)) {
                //     data_vec_port3 = bottom3[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port3-padding)*data_dim1 + (feature_idx_dim1-padding)];
                // } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                //             // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                //     #pragma unroll
                //     for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                //         data_vec_port3.data[vv] = CZERO;
                //     }
                // }
                // switch(win_itm_y_offset)
                // {
                // case 0:
                //     feature_win_buffer_9[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port3; break;
                // case 1:
                //     feature_win_buffer_10[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port3; break;
                // case 2:
                //     feature_win_buffer_11[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port3; break;
                // }

                // if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port4>=padding && feature_idx_dim2_port4<data_dim2+padding)) {
                //     data_vec_port4 = bottom4[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port4-padding)*data_dim1 + (feature_idx_dim1-padding)];
                // } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                //             // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                //     #pragma unroll
                //     for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                //         data_vec_port4.data[vv] = CZERO;
                //     }
                // }
                // switch(win_itm_y_offset)
                // {
                // case 0:
                //     feature_win_buffer_12[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port4; break;
                // case 1:
                //     feature_win_buffer_13[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port4; break;
                // case 2:
                //     feature_win_buffer_14[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port4; break;
                // }

#ifdef DEBUG_MEMRD
                // if(gp_num_x_winbuf==0 && gp_num_y_winbuf==1 && out_idx_z_winbuf==0 && win_itm_y_offset==0 && win_itm_x<10){
                // printf("work-item x=%d, y=%d, z=%d, offset=%d, write data in channel 0=%f\n", win_itm_x, win_itm_y_port, win_itm_z, data_offset, (float)feature_win_buffer_0[(~flag)&0x01][feature_win_buffer_rd_addr].data[0]);
                // printf("work-item x=%d, y=%d, z=%d, write weight in channel 0=%f\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, (float)weight_ch_vec.data[0].lane[0].data[0]);
                // }
                if(gp_num_x_winbuf==0 && gp_num_y_winbuf<=2 && out_idx_z_winbuf==0){
                printf("x=%d\ty=%d\tz=%d\ty_cyc=%d\ty_off=%d\n", win_itm_x, win_itm_y_port, win_itm_z, win_itm_y_cycling, win_itm_y_offset);
                // printf("win_itm_z=%d\tgp_x=%d\tgp_y=%d\tgp_z=%d\tgp_win_x=%d\tgp_win_y=%d\tgp_win_z=%d\n", win_itm_z, gp_num_x, gp_num_y, out_idx_z, gp_num_x_winbuf, gp_num_y_winbuf, out_idx_z_winbuf);
                }
#endif

                // selecting write port
                // z dim        
		        if((win_itm_y_port==PE_NUM_Y_DIV-1) && (win_itm_x==win_size_x-1))
		        	win_itm_y_offset = 0;
		        else if((win_itm_x==win_size_x-1) && (win_itm_y_cycling==stride-1))
		        	win_itm_y_offset++;

                // stride         
		        if(((win_itm_y_cycling==stride-1)||(win_itm_y_port==PE_NUM_Y_DIV-1)) && (win_itm_x==win_size_x-1))
		        	win_itm_y_cycling = 0;
		        else if(win_itm_x==win_size_x-1)
		        	win_itm_y_cycling++;

                // used as virtual loop counters
                // y dim        
                if((win_itm_y_port==PE_NUM_Y_DIV-1) && (win_itm_x==win_size_x-1)){ 
                    win_itm_y_port = 0;
                    win_itm_z++;
                }else if(win_itm_x==win_size_x-1){
                    win_itm_y_port++;
                }
                // x dim        
                if(win_itm_x==win_size_x-1){
                    win_itm_x = 0;
                }else{
                    win_itm_x++;
                }
            }

            // Load weight into weight buffer
            // in time of the first item                                       
            if(gp_item_idx_x==0 && out_idx_xyz>0) {

                weight_ch_tmp = weights[out_idx_z*weight_dim1x2x3/VEC_SIZE + output_idx_dim3*weight_dim1x2 + output_idx_dim2*weight_dim1 + output_idx_dim1];
#ifdef DEF_BUF_WIDTH
                weight_addr = output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1;
                WRITE_BUF(weight_buffer, weight_addr, weight_ch_tmp);
#else                
                weight_buffer[output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1] = weight_ch_tmp;
#endif

            }

            // Only output data for valid convolution work-items
            // In this version, grouping is only performed in row (x) direction
            // if(gp_num_x*gp_size_x+gp_item_idx_x<conv_x && out_idx_xyz>0) {
            if(gp_item_idx_x<win_item && out_idx_xyz>0) {

                if(output_idx_dim1==0 && output_idx_dim2==0 && output_idx_dim3==0) {
                    bias_ch_in = bias[out_idx_z];
                    write_channel_intel(bias_ch, bias_ch_in);

                    // #ifdef DEBUG_MEMRD
                    //printf("work-item x=%d, y=%d, z=%d, channel =0, write bias=%d\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, bias_ch_in.lane[0]);
                    //#endif
                }

                // data
                // feature_y_parallel = win_buffer[flag][output_idx_dim3*win_size_y*win_size_x + output_idx_dim2*win_size_x + (output_idx_dim1+gp_item_idx_x*stride)];

#ifdef STRIDE_1
		        if(stride==1){
		        	idx_y_cycling = 0;
		        	idx_y_offset = output_idx_dim2;
		        	feature_win_buffer_wr_addr = output_idx_dim3*win_size_x + output_idx_dim1+gp_item_idx_x;
		        }
#endif
#ifdef STRIDE_4		
		        else if(stride==4){
		        	idx_y_cycling = output_idx_dim2&0x03;
		        	idx_y_offset = output_idx_dim2>>2;
		        	feature_win_buffer_wr_addr = output_idx_dim3*4*win_size_x + idx_y_cycling*win_size_x + output_idx_dim1+gp_item_idx_x*4;
		        }
#endif	

                feature_y_parallel[0] = feature_win_buffer_0[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[1] = feature_win_buffer_1[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[2] = feature_win_buffer_2[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[3] = feature_win_buffer_3[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[4] = feature_win_buffer_4[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[5] = feature_win_buffer_5[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[6] = feature_win_buffer_6[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[7] = feature_win_buffer_7[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[8] = feature_win_buffer_8[flag][feature_win_buffer_wr_addr];
                // for padding when weight_w is 3
                feature_y_parallel[9] = feature_win_buffer_9[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[10] = feature_win_buffer_10[flag][feature_win_buffer_wr_addr];
#if PE_NUM_Y>9
                feature_y_parallel[11] = feature_win_buffer_11[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[12] = feature_win_buffer_12[flag][feature_win_buffer_wr_addr];
#endif
#if PE_NUM_Y>11
                feature_y_parallel[13] = feature_win_buffer_13[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[14] = feature_win_buffer_14[flag][feature_win_buffer_wr_addr];
#endif
#if PE_NUM_Y>13
                feature_y_parallel[15] = feature_win_buffer_15[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[16] = feature_win_buffer_16[flag][feature_win_buffer_wr_addr];
#endif
#if PE_NUM_Y>15
                feature_y_parallel[17] = feature_win_buffer_17[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[18] = feature_win_buffer_18[flag][feature_win_buffer_wr_addr];
#endif

                switch(idx_y_offset)
		        {

                    case 0:
			        #pragma unroll
			        for(uchar yy=0; yy<PE_NUM_Y; yy++){feature_y.data[yy] = feature_y_parallel[0+yy];} break;
			        case 1:
			        #pragma unroll
			        for(uchar yy=0; yy<PE_NUM_Y; yy++){feature_y.data[yy] = feature_y_parallel[1+yy];} break;
			        case 2: // to support all layers (kernel size=3, stride=1)
			        #pragma unroll
			        for(uchar yy=0; yy<PE_NUM_Y; yy++){feature_y.data[yy] = feature_y_parallel[2+yy];} break;
#ifndef YOLO        
                    case 3: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[3]; break;
			        case 4: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[4]; break;
			        case 5: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[5]; break;
			        case 6: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[6]; break;
			        case 7: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[7]; break;
			        case 8: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[8]; break;
			        case 9: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[9]; break;
			        case 10: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[10]; break;
			        case 11: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[11]; break;
			        case 12: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[12]; break;
			        case 13: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[13]; break;
			        case 14: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[14]; break;
			        // case 15: // for FC layer, only read single line
			        // feature_y.data[0] = feature_y_parallel[15]; break;
#endif
                }

#ifdef CHANNEL_OPT
                write_channel_intel(data_ch, feature_y);
#else
                #pragma unroll
                for(unsigned char ll=0; ll<LANE_NUM; ll++) {
                    data_ch_vec.lane[ll] = feature_y;
                }
                write_channel_intel(data_ch, data_ch_vec);
#endif

                // weight and bias fetcher
#ifdef DEF_BUF_WIDTH
                weight_addr = output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1;
                READ_BUF(weight_buffer, weight_addr, weight_ch_tmp);
#else
                weight_ch_tmp = weight_buffer[output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1];
                //weight_ch_tmp = weights[out_idx_z*weight_dim1x2x3/VEC_SIZE + output_idx_dim3*weight_dim1x2 + output_idx_dim2*weight_dim1 + output_idx_dim1];
#endif 
#ifdef CHANNEL_OPT 
                write_channel_intel(weight_ch, weight_ch_tmp);
#else      
                #pragma unroll
				for(unsigned char yy=0; yy<PE_NUM_Y; yy++){
					weight_ch_vec.data[yy] = weight_ch_tmp;
				}
                write_channel_intel(weight_ch, weight_ch_vec);
#endif

#ifdef DEBUG_MEMRD
                // if(gp_num_x==0 && gp_num_y==1 && out_idx_z==0 && gp_item_idx_x==0){
                // printf("work-item x=%d, y=%d, z=%d, offset=%d, write data in channel 0=%f\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, data_offset, (float)data_ch_vec.lane[0].data[0].data[0]);
                // printf("work-item x=%d, y=%d, z=%d, write weight in channel 0=%f\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, (float)weight_ch_vec.data[0].lane[0].data[0]);
                // printf("%d\t", feature_win_buffer_0[flag][feature_win_buffer_wr_addr].data[0]);
                // }

                if(gp_num_y==0 && out_idx_z==0 && output_idx_dim2==0 && output_idx_dim1==0 && output_idx_dim3==0)
                // printf("out_dim1=%d\tout_dim2=%d\tout_dim3=%d\titm_idx=%d\tclcying=%d\toffset=%d\n",
                //         output_idx_dim1, output_idx_dim2, output_idx_dim3, gp_item_idx_x, idx_y_cycling, idx_y_offset);
                printf("gp_num_x=%d\tgp_size_x=%d\tgp_item_idx_x=%d\ttotal=%d\n", gp_num_x, gp_size_x, gp_item_idx_x, gp_num_x*gp_size_x+gp_item_idx_x);
#endif

                // used as output loop counters
                if((output_idx_dim3==weight_dim3/VEC_SIZE-1) && (output_idx_dim2==weight_dim2-1) && (output_idx_dim1==weight_dim1-1)) {
                    output_idx_dim3 = 0;
                    gp_item_idx_x++;
                } else if((output_idx_dim2==weight_dim2-1)&& (output_idx_dim1==weight_dim1-1))
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

        }

        //		}// end of win_itm_z
        //	}// end of win_itm_y
        //}// end of win_itm_x

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

        // used as virtual group loop counters
        if((out_idx_z==weight_dim4_div_lane-1) && (gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
            out_idx_z = 0;
        else if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
            out_idx_z++;

        if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
            gp_num_y = 0;
        else if(gp_num_x==group_num_x-1)
            gp_num_y++;

        if(gp_num_x==group_num_x-1)
            gp_num_x = 0;
        else
            gp_num_x++;

        // if(out_idx_z>=6)
        // printf("gp_x=%d\tgp_y=%d\tgp_z=%d\tgp_win_x=%d\tgp_win_y=%d\tgp_win_z=%d\n", gp_num_x, gp_num_y, out_idx_z, gp_num_x_winbuf, gp_num_y_winbuf, out_idx_z_winbuf);


        //			}// end of gp_num_x
        //		}// end of gp_num_y
        //}// end of out_idx_z
    }

    //printf("Kernel 0 lanched !!!\n");
}


#elif defined RD_MULTPORT_4_3  // for PE_NUM_Y=9
// Fetch Data from Global Memory
__kernel
__attribute__((task))
__attribute__((max_global_work_dim(0)))
void memRead(
    // Params Ports
    ushort  data_dim1,
    ushort  data_dim2,
    uint data_dim1xdim2,
    uchar  weight_dim1,
    uchar  weight_dim2,
    ushort weight_dim3,
    ushort weight_dim4_div_lane, // avoid generating divider
    uchar  weight_dim1x2,
    uint   weight_dim1x2x3,
    ushort  conv_x,
    //uchar  conv_y,           // not used in this version
    uchar  stride,
    uchar  padding,
    uchar  split,
    uchar gp_size_y,
    ushort gp_last_size_x,
    ushort gp_size_x,
    ushort  group_num_x,
    ushort  group_num_y,
    uint  conv_group_num_dim1x2_dim4_div_lane,
    // uchar  group_rem_size_x,
    //uchar  group_rem_size_y, // not used in this version
    // uint   group_rem_size_xyz,
    ushort win_size_x,
    // uchar  win_size_y_port,
    uint   win_size_xyz_port,
    uint Item_Loop_Bound,
    uint Item_Last_Loop_Bound,
    // Data Ports
    __global DPTYPE_VEC    *restrict bottom0,
    __global DPTYPE_VEC    *restrict bottom1,
    __global DPTYPE_VEC    *restrict bottom2,
    __global DPTYPE_VEC    *restrict bottom3,
    // __global DPTYPE_VEC    *restrict bottom4,
    __global DPTYPE_SCAL_VEC  *restrict weights,
    __global volatile DPTYPE_SCAL *restrict bias        
){

    // feature win buffer
    // Ping-pong buffer
    __local DPTYPE_VEC  feature_win_buffer_0[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_1[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_2[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_3[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_4[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_5[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_6[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_7[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_8[2][WIN_BUF_DEPTH];
    // for padding when weight_w is 3
    __local DPTYPE_VEC  feature_win_buffer_9[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_10[2][WIN_BUF_DEPTH];


    // Weight buffer
#ifdef DEF_BUF_WIDTH
    __local DPTYPE_VEC DEFINE_BUF(weight_buffer, WEIGHT_BUF_SIZE);
#else
    __local DPTYPE_SCAL_VEC  weight_buffer[WEIGHT_BUF_SIZE];
#endif

    // Input Data, Weights and Bias
    // read from feature_win_buffer_0 (1,2,3,...,14)
    DPTYPE_VEC    feature_y_parallel[PE_NUM_Y+WIN_BUF_Y_PAD];
    // take data of PE_NUM_Y from feature_y_parallel
    // DPTYPE_PE_VEC    feature_y;
    // read from bottom
    DPTYPE_VEC    data_vec_port0;
    DPTYPE_VEC    data_vec_port1;
    DPTYPE_VEC    data_vec_port2;
    DPTYPE_VEC    data_vec_port3;
    // DPTYPE_VEC    data_vec_port4;
    // write to kl_conv
    SCAL_PE_VEC   data_ch_vec;
    DPTYPE_SCAL_VEC weight_ch_tmp;
    PE_SCAL_VEC   weight_ch_vec;
    DPTYPE_SCAL  bias_ch_in;

    // feature win buffer read/write address
    // ushort     feature_win_buffer_rd_addr, feature_win_buffer_wr_addr;
    ushort     data_offset = 0; // assuming the 1st layer is not in split
    uchar  flag; // ping-pong flag

    // virtual loop counters
    // for loading feature to feature_win_buffer from bottom
    ushort win_itm_z;

	uchar  win_itm_y_port;
	// uchar  win_itm_y_cycling, win_itm_y_offset;
    // for transfer data to knl_conv from feature_win_buffer
    ushort output_idx_dim3;
    uchar  output_idx_dim1, output_idx_dim2;
    // uchar  idx_y_cycling, idx_y_offset;
    // for group counters
    short gp_num_x;
    ushort gp_num_y, out_idx_z;
    ushort gp_num_x_winbuf, gp_num_y_winbuf, out_idx_z_winbuf;

    // for global index
    ushort feature_idx_dim1;  // for yolov2, maximum is 418(padding)
    ushort feature_idx_dim2_port0; // for yolov2, maximum is 418(padding)
    ushort feature_idx_dim2_port1;
    ushort feature_idx_dim2_port2;
    ushort feature_idx_dim2_port3;
    ushort feature_idx_dim2_port4;
    ushort feature_idx_dim3;
    // counter the number of x-dimensional convolutions in the feature_win_buffer
    ushort  gp_item_idx_x;
    ushort  win_itm_x;
    ushort  win_item; 
    uint   item_loop_bound;


    gp_num_x_winbuf = 0; // there is only one group for FC mode when batch=1
    gp_num_y_winbuf = 0;
    out_idx_z_winbuf = 0;

    // reset global group virtual loop counters
    gp_num_x = -1;
    gp_num_y = 0;
    out_idx_z = 0;

#ifdef DEF_BUF_WIDTH
#pragma ivdep array(weight_buffer_0)
#pragma ivdep array(weight_buffer_1)
#pragma ivdep array(weight_buffer_2)
#else
#pragma ivdep array(weight_buffer)
#endif                                                                   
Group:
    //printf("%d\n", weight_dim4_div_lane*group_num_y*group_num_x);
    //printf("%d\t%d\t%d\n", weight_dim4_div_lane, group_num_y, group_num_x);
    // printf("memRd: %d\n", conv_group_num_dim1x2_dim4_div_lane);

    for(unsigned int out_idx_xyz=0; out_idx_xyz<=conv_group_num_dim1x2_dim4_div_lane; out_idx_xyz++) {
        // The following group loops are flattened as the upper loop to improve pipeline efficiency
        //for(unsigned short out_idx_z=0; out_idx_z<weight_dim4_div_lane; out_idx_z++){
        
        // special case when split==1, the output feature maps depend on only half the input feature maps
        if(split==0)
            data_offset = 0;
        else if(out_idx_z_winbuf<(weight_dim4_div_lane>>1)) // the lower half of the output feature maps depend on the lower half of the input
            data_offset = 0;
        else
            data_offset = weight_dim3/VEC_SIZE;	// the upper half of the output feature maps depend on the upper half of the input

        //for(unsigned short gp_num_y=0; gp_num_y<group_num_y; gp_num_y++){
        //for(unsigned short gp_num_x=0; gp_num_x<group_num_x+1; gp_num_x++){ // add one more extra iteration for ping-pong buffering operations

        flag = out_idx_xyz & 0x01; //ping-pong flag
        
        // feature win buffer read/write address
        ushort     feature_win_buffer_rd_addr, feature_win_buffer_wr_addr;
#ifdef DEF_BUF_WIDTH
        ushort     weight_addr;
#endif                               
        uchar  win_itm_y_cycling, win_itm_y_offset;
        uchar  idx_y_cycling, idx_y_offset;
        // PE_SCAL_VEC    feature_y;
        DPTYPE_PE_VEC feature_y;

        // reset output loop counters
        output_idx_dim1 = 0;
        output_idx_dim2 = 0;
        output_idx_dim3 = 0;
        // reset in-group item counters
        gp_item_idx_x = 0;
        
        win_itm_y_offset = 0;

        // reset input winbuffer loop counters
        win_itm_x = 0;
        win_itm_y_port = 0;
        win_itm_z = 0;


        if(gp_num_x==group_num_x-1) {// last group in each row
            // ensuring that both winbuf load loop and output loop are finished, i.e., use a larger value as the loop bound
            // item_loop_bound = win_size_x>=group_rem_size_x?(win_size_xyz/VEC_SIZE):(group_rem_size_xyz/VEC_SIZE);
            item_loop_bound = Item_Last_Loop_Bound;
            win_item = gp_last_size_x;
        }else {
            // if(stride>=weight_dim1 || stride>=weight_dim2) // special case convolution layers with stride>weight_dim1/2, such as resnet50
            //     item_loop_bound = win_size_xyz/VEC_SIZE;
            // else
                // item_loop_bound = (weight_dim1x2x3*CONV_GP_SIZE_Y*CONV_GP_SIZE_X/VEC_SIZE);
                item_loop_bound = Item_Loop_Bound;
                win_item = gp_size_x;
        }

#pragma ivdep array(feature_win_buffer_0)
#pragma ivdep array(feature_win_buffer_1)
#pragma ivdep array(feature_win_buffer_2)
#pragma ivdep array(feature_win_buffer_3)
#pragma ivdep array(feature_win_buffer_4)
#pragma ivdep array(feature_win_buffer_5)
#pragma ivdep array(feature_win_buffer_6)
#pragma ivdep array(feature_win_buffer_7)
#pragma ivdep array(feature_win_buffer_8)
#pragma ivdep array(feature_win_buffer_9)
#pragma ivdep array(feature_win_buffer_10)
#ifdef DEF_BUF_WIDTH
#pragma ivdep array(weight_buffer_0)
#pragma ivdep array(weight_buffer_1)
#pragma ivdep array(weight_buffer_2)
#else
#pragma ivdep array(weight_buffer)
#endif 
Item:
        for(unsigned int  win_itm_xyz=0; win_itm_xyz<item_loop_bound; win_itm_xyz++) {
            //// The following loops are flattened as the upper loop to improve pipeline efficiency
            //for(unsigned short win_itm_z=0; win_itm_z<weight_dim3/VEC_SIZE; win_itm_z++){
            //	for(unsigned char  win_itm_y=0; win_itm_y<weight_dim2*CONV_GP_SIZE_Y; win_itm_y++){
            //		for(unsigned char  win_itm_x=0; win_itm_x<weight_dim1*CONV_GP_SIZE_X; win_itm_x++){

            // Winbuffer loading operations
            if(win_itm_z<(weight_dim3/VEC_SIZE) && out_idx_xyz<conv_group_num_dim1x2_dim4_div_lane) {
            
#ifdef STRIDE_1
		        if(stride==1){
		        	feature_win_buffer_rd_addr = win_itm_z*win_size_x + win_itm_x;
		        }
#endif
#ifdef STRIDE_4
		        else if(stride==4){
		        	feature_win_buffer_rd_addr = win_itm_z*win_size_x*4 + win_itm_y_cycling*win_size_x + win_itm_x;
		        }
#endif

                feature_idx_dim1 = win_itm_x + gp_num_x_winbuf*gp_size_x*stride;
                feature_idx_dim2_port0 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride;
                feature_idx_dim2_port1 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride + PE_NUM_Y_DIV;
                feature_idx_dim2_port2 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride + PE_NUM_Y_DIV*2;
                feature_idx_dim2_port3 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride + PE_NUM_Y_DIV*3;
                feature_idx_dim2_port4 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride + PE_NUM_Y_DIV*4;
                feature_idx_dim3 = win_itm_z;

                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port0>=padding && feature_idx_dim2_port0<data_dim2+padding)) {
                    data_vec_port0 = bottom0[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port0-padding)*data_dim1 + (feature_idx_dim1-padding)];
                } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                            // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                    #pragma unroll
                    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                        data_vec_port0.data[vv] = CZERO;
                    }
                }
                switch(win_itm_y_offset)
                {
                case 0:
                    feature_win_buffer_0[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 1:
                    feature_win_buffer_1[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 2:
                    feature_win_buffer_2[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                // case 3:
                //     feature_win_buffer_3[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                // case 4:
                //     feature_win_buffer_4[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                }

                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port1>=padding && feature_idx_dim2_port1<data_dim2+padding)) {
                    data_vec_port1 = bottom1[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port1-padding)*data_dim1 + (feature_idx_dim1-padding)];
                } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                            // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                    #pragma unroll
                    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                        data_vec_port1.data[vv] = CZERO;
                    }
                }
                switch(win_itm_y_offset)
                {
                case 0:
                    feature_win_buffer_3[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                case 1:
                    feature_win_buffer_4[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                case 2:
                    feature_win_buffer_5[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                // case 3:
                //     feature_win_buffer_8[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                // case 4:
                //     feature_win_buffer_9[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                }

                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port2>=padding && feature_idx_dim2_port2<data_dim2+padding)) {
                    data_vec_port2 = bottom2[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port2-padding)*data_dim1 + (feature_idx_dim1-padding)];
                } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                            // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                    #pragma unroll
                    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                        data_vec_port2.data[vv] = CZERO;
                    }
                }
                // if((win_item_y_port+PE_NUM_Y_DIV*2)<gp_size_y) //for 1x1 conv, don't need padding
                switch(win_itm_y_offset)
                {
                case 0:
                    feature_win_buffer_6[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                case 1:
                    feature_win_buffer_7[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                case 2:
                    feature_win_buffer_8[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                // case 3:
                //     feature_win_buffer_13[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                // case 4:
                //     feature_win_buffer_14[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                }

                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port3>=padding && feature_idx_dim2_port3<data_dim2+padding)) {
                    data_vec_port3 = bottom3[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port3-padding)*data_dim1 + (feature_idx_dim1-padding)];
                } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                            // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                    #pragma unroll
                    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                        data_vec_port3.data[vv] = CZERO;
                    }
                }
                switch(win_itm_y_offset)
                {
                case 0:
                    feature_win_buffer_9[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port3; break;
                case 1:
                    feature_win_buffer_10[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port3; break;
                // case 2:
                //     feature_win_buffer_11[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port3; break;
                default:
                    break;
                }


#ifdef DEBUG_MEMRD
                // if(gp_num_x_winbuf==0 && gp_num_y_winbuf==1 && out_idx_z_winbuf==0 && win_itm_y_offset==0 && win_itm_x<10){
                // printf("work-item x=%d, y=%d, z=%d, offset=%d, write data in channel 0=%f\n", win_itm_x, win_itm_y_port, win_itm_z, data_offset, (float)feature_win_buffer_0[(~flag)&0x01][feature_win_buffer_rd_addr].data[0]);
                // printf("work-item x=%d, y=%d, z=%d, write weight in channel 0=%f\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, (float)weight_ch_vec.data[0].lane[0].data[0]);
                // }
                if(gp_num_x_winbuf==0 && gp_num_y_winbuf<=2 && out_idx_z_winbuf==0){
                printf("x=%d\ty=%d\tz=%d\ty_cyc=%d\ty_off=%d\n", win_itm_x, win_itm_y_port, win_itm_z, win_itm_y_cycling, win_itm_y_offset);
                // printf("win_itm_z=%d\tgp_x=%d\tgp_y=%d\tgp_z=%d\tgp_win_x=%d\tgp_win_y=%d\tgp_win_z=%d\n", win_itm_z, gp_num_x, gp_num_y, out_idx_z, gp_num_x_winbuf, gp_num_y_winbuf, out_idx_z_winbuf);
                }
#endif

                // selecting write port
		        if((win_itm_y_port==PE_NUM_Y_DIV-1) && (win_itm_x==win_size_x-1))
		        	win_itm_y_offset = 0;
		        else if((win_itm_x==win_size_x-1) && (win_itm_y_cycling==stride-1))
		        	win_itm_y_offset++;

		        if(((win_itm_y_cycling==stride-1)||(win_itm_y_port==PE_NUM_Y_DIV-1)) && (win_itm_x==win_size_x-1))
		        	win_itm_y_cycling = 0;
		        else if(win_itm_x==win_size_x-1)
		        	win_itm_y_cycling++;

                // used as virtual loop counters
                if((win_itm_y_port==PE_NUM_Y_DIV-1) && (win_itm_x==win_size_x-1)){ 
                    win_itm_y_port = 0;
                    win_itm_z++;
                }else if(win_itm_x==win_size_x-1){
                    win_itm_y_port++;
                }
                if(win_itm_x==win_size_x-1){
                    win_itm_x = 0;
                }else{
                    win_itm_x++;
                }
            }

            // Load weight into weight buffer
            if(gp_item_idx_x==0 && out_idx_xyz>0) {

                weight_ch_tmp = weights[out_idx_z*weight_dim1x2x3/VEC_SIZE + output_idx_dim3*weight_dim1x2 + output_idx_dim2*weight_dim1 + output_idx_dim1];
#ifdef DEF_BUF_WIDTH
                weight_addr = output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1;
                WRITE_BUF(weight_buffer, weight_addr, weight_ch_tmp);
#else                
                weight_buffer[output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1] = weight_ch_tmp;
#endif                                       
            }

            // Only output data for valid convolution work-items
            // In this version, grouping is only performed in row (x) direction
            // if(gp_num_x*gp_size_x+gp_item_idx_x<conv_x && out_idx_xyz>0) {
            if(gp_item_idx_x<win_item && out_idx_xyz>0) {

                if(output_idx_dim1==0 && output_idx_dim2==0 && output_idx_dim3==0) {
                    bias_ch_in = bias[out_idx_z];
                    write_channel_intel(bias_ch, bias_ch_in);

                    // #ifdef DEBUG_MEMRD
                    //printf("work-item x=%d, y=%d, z=%d, channel =0, write bias=%d\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, bias_ch_in.lane[0]);
                    //#endif
                }

                // data
                // feature_y_parallel = win_buffer[flag][output_idx_dim3*win_size_y*win_size_x + output_idx_dim2*win_size_x + (output_idx_dim1+gp_item_idx_x*stride)];

#ifdef STRIDE_1
		        if(stride==1){
		        	idx_y_cycling = 0;
		        	idx_y_offset = output_idx_dim2;
		        	feature_win_buffer_wr_addr = output_idx_dim3*win_size_x + output_idx_dim1+gp_item_idx_x;
		        }
#endif
#ifdef STRIDE_4		
		        else if(stride==4){
		        	idx_y_cycling = output_idx_dim2&0x03;
		        	idx_y_offset = output_idx_dim2>>2;
		        	feature_win_buffer_wr_addr = output_idx_dim3*4*win_size_x + idx_y_cycling*win_size_x + output_idx_dim1+gp_item_idx_x*4;
		        }
#endif	

                feature_y_parallel[0] = feature_win_buffer_0[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[1] = feature_win_buffer_1[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[2] = feature_win_buffer_2[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[3] = feature_win_buffer_3[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[4] = feature_win_buffer_4[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[5] = feature_win_buffer_5[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[6] = feature_win_buffer_6[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[7] = feature_win_buffer_7[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[8] = feature_win_buffer_8[flag][feature_win_buffer_wr_addr];
                // for padding when weight_w is 3
                feature_y_parallel[9] = feature_win_buffer_9[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[10] = feature_win_buffer_10[flag][feature_win_buffer_wr_addr];


                switch(idx_y_offset)
		        {
                    case 0:
			        #pragma unroll
			        for(uchar yy=0; yy<PE_NUM_Y; yy++){feature_y.data[yy] = feature_y_parallel[0+yy];} break;
			        case 1:
			        #pragma unroll
			        for(uchar yy=0; yy<PE_NUM_Y; yy++){feature_y.data[yy] = feature_y_parallel[1+yy];} break;
			        case 2: // to support all layers (kernel size=3, stride=1)
			        #pragma unroll
			        for(uchar yy=0; yy<PE_NUM_Y; yy++){feature_y.data[yy] = feature_y_parallel[2+yy];} break;
#ifndef YOLO        
                    case 3: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[3]; break;
			        case 4: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[4]; break;
			        case 5: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[5]; break;
			        case 6: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[6]; break;
			        case 7: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[7]; break;
			        case 8: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[8]; break;
			        case 9: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[9]; break;
			        case 10: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[10]; break;
			        case 11: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[11]; break;
			        case 12: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[12]; break;
			        case 13: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[13]; break;
			        case 14: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[14]; break;
			        // case 15: // for FC layer, only read single line
			        // feature_y.data[0] = feature_y_parallel[15]; break;
#endif
                }

#ifdef CHANNEL_OPT
                write_channel_intel(data_ch, feature_y);
#else
                #pragma unroll
                for(unsigned char ll=0; ll<LANE_NUM; ll++) {
                    data_ch_vec.lane[ll] = feature_y;
                }
                write_channel_intel(data_ch, data_ch_vec);
#endif

                // weight and bias fetcher
#ifdef DEF_BUF_WIDTH
                weight_addr = output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1;
                READ_BUF(weight_buffer, weight_addr, weight_ch_tmp);
#else
                weight_ch_tmp = weight_buffer[output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1];
                //weight_ch_tmp = weights[out_idx_z*weight_dim1x2x3/VEC_SIZE + output_idx_dim3*weight_dim1x2 + output_idx_dim2*weight_dim1 + output_idx_dim1];
#endif                                                              
#ifdef CHANNEL_OPT 
                write_channel_intel(weight_ch, weight_ch_tmp);
#else      
                #pragma unroll
				for(unsigned char yy=0; yy<PE_NUM_Y; yy++){
					weight_ch_vec.data[yy] = weight_ch_tmp;
				}
                write_channel_intel(weight_ch, weight_ch_vec);
#endif

#ifdef DEBUG_MEMRD
                // if(gp_num_x==0 && gp_num_y==1 && out_idx_z==0 && gp_item_idx_x==0){
                // printf("work-item x=%d, y=%d, z=%d, offset=%d, write data in channel 0=%f\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, data_offset, (float)data_ch_vec.lane[0].data[0].data[0]);
                // printf("work-item x=%d, y=%d, z=%d, write weight in channel 0=%f\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, (float)weight_ch_vec.data[0].lane[0].data[0]);
                // printf("%d\t", feature_win_buffer_0[flag][feature_win_buffer_wr_addr].data[0]);
                // }

                if(gp_num_y==0 && out_idx_z==0 && output_idx_dim2==0 && output_idx_dim1==0 && output_idx_dim3==0)
                // printf("out_dim1=%d\tout_dim2=%d\tout_dim3=%d\titm_idx=%d\tclcying=%d\toffset=%d\n",
                //         output_idx_dim1, output_idx_dim2, output_idx_dim3, gp_item_idx_x, idx_y_cycling, idx_y_offset);
                printf("gp_num_x=%d\tgp_size_x=%d\tgp_item_idx_x=%d\ttotal=%d\n", gp_num_x, gp_size_x, gp_item_idx_x, gp_num_x*gp_size_x+gp_item_idx_x);
#endif

                // used as output loop counters
                if((output_idx_dim3==weight_dim3/VEC_SIZE-1) && (output_idx_dim2==weight_dim2-1) && (output_idx_dim1==weight_dim1-1)) {
                    output_idx_dim3 = 0;
                    gp_item_idx_x++;
                } else if((output_idx_dim2==weight_dim2-1)&& (output_idx_dim1==weight_dim1-1))
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

        }

        //		}// end of win_itm_z
        //	}// end of win_itm_y
        //}// end of win_itm_x

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

        // used as virtual group loop counters
        if((out_idx_z==weight_dim4_div_lane-1) && (gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
            out_idx_z = 0;
        else if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
            out_idx_z++;

        if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
            gp_num_y = 0;
        else if(gp_num_x==group_num_x-1)
            gp_num_y++;

        if(gp_num_x==group_num_x-1)
            gp_num_x = 0;
        else
            gp_num_x++;

        // if(out_idx_z>=6)
        // printf("gp_x=%d\tgp_y=%d\tgp_z=%d\tgp_win_x=%d\tgp_win_y=%d\tgp_win_z=%d\n", gp_num_x, gp_num_y, out_idx_z, gp_num_x_winbuf, gp_num_y_winbuf, out_idx_z_winbuf);


        //			}// end of gp_num_x
        //		}// end of gp_num_y
        //}// end of out_idx_z
    }

    //printf("Kernel 0 lanched !!!\n");
}



#elif defined RD_MULTPORT_5_3  // for PE_NUM_Y = 11 / 13
// Fetch Data from Global Memory
__kernel
__attribute__((task))
__attribute__((max_global_work_dim(0)))
void memRead(
    // Params Ports
    ushort  data_dim1,
    ushort  data_dim2,
    uint data_dim1xdim2,
    uchar  weight_dim1,
    uchar  weight_dim2,
    ushort weight_dim3,
    ushort weight_dim4_div_lane, // avoid generating divider
    uchar  weight_dim1x2,
    uint   weight_dim1x2x3,
    ushort  conv_x,
    //uchar  conv_y,           // not used in this version
    uchar  stride,
    uchar  padding,
    uchar  split,
    uchar gp_size_y,
    ushort gp_last_size_x,
    ushort gp_size_x,
    ushort  group_num_x,
    ushort  group_num_y,
    uint  conv_group_num_dim1x2_dim4_div_lane,
    // uchar  group_rem_size_x,
    //uchar  group_rem_size_y, // not used in this version
    // uint   group_rem_size_xyz,
    ushort win_size_x,
    // uchar  win_size_y_port,
    uint   win_size_xyz_port,
    uint Item_Loop_Bound,
    uint Item_Last_Loop_Bound,
    // Data Ports
    __global DPTYPE_VEC    *restrict bottom0,
    __global DPTYPE_VEC    *restrict bottom1,
    __global DPTYPE_VEC    *restrict bottom2,
    __global DPTYPE_VEC    *restrict bottom3,
    __global DPTYPE_VEC    *restrict bottom4,
    __global DPTYPE_SCAL_VEC  *restrict weights,
    __global volatile DPTYPE_SCAL *restrict bias        
){

    // feature win buffer
    // Ping-pong buffer
    __local DPTYPE_VEC  feature_win_buffer_0[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_1[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_2[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_3[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_4[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_5[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_6[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_7[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_8[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_9[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_10[2][WIN_BUF_DEPTH];
    // for padding when weight_w is 3
    __local DPTYPE_VEC  feature_win_buffer_11[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_12[2][WIN_BUF_DEPTH];
#if PE_NUM_Y>11
    __local DPTYPE_VEC  feature_win_buffer_13[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_14[2][WIN_BUF_DEPTH];
#endif

    // Weight buffer
#ifdef DEF_BUF_WIDTH
    __local DPTYPE_VEC DEFINE_BUF(weight_buffer, WEIGHT_BUF_SIZE);
#else
    __local DPTYPE_SCAL_VEC  weight_buffer[WEIGHT_BUF_SIZE];
#endif

    // Input Data, Weights and Bias
    // read from feature_win_buffer_0 (1,2,3,...,14)
    DPTYPE_VEC    feature_y_parallel[PE_NUM_Y+WIN_BUF_Y_PAD];
    // take data of PE_NUM_Y from feature_y_parallel
    // DPTYPE_PE_VEC    feature_y;
    // read from bottom
    DPTYPE_VEC    data_vec_port0;
    DPTYPE_VEC    data_vec_port1;
    DPTYPE_VEC    data_vec_port2;
    DPTYPE_VEC    data_vec_port3;
    DPTYPE_VEC    data_vec_port4;
    // write to kl_conv
    SCAL_PE_VEC   data_ch_vec;
    DPTYPE_SCAL_VEC weight_ch_tmp;
    PE_SCAL_VEC   weight_ch_vec;
    DPTYPE_SCAL  bias_ch_in;

    // feature win buffer read/write address
    // ushort     feature_win_buffer_rd_addr, feature_win_buffer_wr_addr;
    ushort     data_offset = 0; // assuming the 1st layer is not in split
    uchar  flag; // ping-pong flag

    // virtual loop counters
    // for loading feature to feature_win_buffer from bottom
    ushort win_itm_z;

	uchar  win_itm_y_port;
	// uchar  win_itm_y_cycling, win_itm_y_offset;
    // for transfer data to knl_conv from feature_win_buffer
    ushort output_idx_dim3;
    uchar  output_idx_dim1, output_idx_dim2;
    // uchar  idx_y_cycling, idx_y_offset;
    // for group counters
    short gp_num_x;
    ushort gp_num_y, out_idx_z;
    ushort gp_num_x_winbuf, gp_num_y_winbuf, out_idx_z_winbuf;

    // for global index
    ushort feature_idx_dim1;  // for yolov2, maximum is 418(padding)
    ushort feature_idx_dim2_port0; // for yolov2, maximum is 418(padding)
    ushort feature_idx_dim2_port1;
    ushort feature_idx_dim2_port2;
    ushort feature_idx_dim2_port3;
    ushort feature_idx_dim2_port4;
    ushort feature_idx_dim3;
    // counter the number of x-dimensional convolutions in the feature_win_buffer
    ushort  gp_item_idx_x;
    ushort  win_itm_x;
    ushort  win_item;
    uint   item_loop_bound;

    gp_num_x_winbuf = 0; // there is only one group for FC mode when batch=1
    gp_num_y_winbuf = 0;
    out_idx_z_winbuf = 0;

    // reset global group virtual loop counters
    gp_num_x = -1;
    gp_num_y = 0;
    out_idx_z = 0;

#ifdef DEF_BUF_WIDTH
#pragma ivdep array(weight_buffer_0)
#pragma ivdep array(weight_buffer_1)
#pragma ivdep array(weight_buffer_2)
#else
#pragma ivdep array(weight_buffer)
#endif       
Group:
    //printf("%d\n", weight_dim4_div_lane*group_num_y*group_num_x);
    //printf("%d\t%d\t%d\n", weight_dim4_div_lane, group_num_y, group_num_x);
    // printf("memRd: %d\n", conv_group_num_dim1x2_dim4_div_lane);

    for(unsigned int out_idx_xyz=0; out_idx_xyz<=conv_group_num_dim1x2_dim4_div_lane; out_idx_xyz++) {
        // The following group loops are flattened as the upper loop to improve pipeline efficiency
        //for(unsigned short out_idx_z=0; out_idx_z<weight_dim4_div_lane; out_idx_z++){
        
        // special case when split==1, the output feature maps depend on only half the input feature maps
        if(split==0)
            data_offset = 0;
        else if(out_idx_z_winbuf<(weight_dim4_div_lane>>1)) // the lower half of the output feature maps depend on the lower half of the input
            data_offset = 0;
        else
            data_offset = weight_dim3/VEC_SIZE;	// the upper half of the output feature maps depend on the upper half of the input

        //for(unsigned short gp_num_y=0; gp_num_y<group_num_y; gp_num_y++){
        //for(unsigned short gp_num_x=0; gp_num_x<group_num_x+1; gp_num_x++){ // add one more extra iteration for ping-pong buffering operations

        flag = out_idx_xyz & 0x01; //ping-pong flag
        
        // feature win buffer read/write address
        ushort     feature_win_buffer_rd_addr, feature_win_buffer_wr_addr;
#ifdef DEF_BUF_WIDTH
        ushort     weight_addr;
#endif
        uchar  win_itm_y_cycling, win_itm_y_offset;
        uchar  idx_y_cycling, idx_y_offset;
        // PE_SCAL_VEC    feature_y;
        DPTYPE_PE_VEC feature_y;

        // reset output loop counters
        output_idx_dim1 = 0;
        output_idx_dim2 = 0;
        output_idx_dim3 = 0;
        // reset in-group item counters
        gp_item_idx_x = 0;
        
        win_itm_y_offset = 0;

        // reset input winbuffer loop counters
        win_itm_x = 0;
        win_itm_y_port = 0;
        win_itm_z = 0;


        if(gp_num_x==group_num_x-1) {// last group in each row
            // ensuring that both winbuf load loop and output loop are finished, i.e., use a larger value as the loop bound
            // item_loop_bound = win_size_x>=group_rem_size_x?(win_size_xyz/VEC_SIZE):(group_rem_size_xyz/VEC_SIZE);
            item_loop_bound = Item_Last_Loop_Bound;
            win_item = gp_last_size_x;
        }else {
            // if(stride>=weight_dim1 || stride>=weight_dim2) // special case convolution layers with stride>weight_dim1/2, such as resnet50
            //     item_loop_bound = win_size_xyz/VEC_SIZE;
            // else
                // item_loop_bound = (weight_dim1x2x3*CONV_GP_SIZE_Y*CONV_GP_SIZE_X/VEC_SIZE);
                item_loop_bound = Item_Loop_Bound;
                win_item = gp_size_x;
        }

#pragma ivdep array(feature_win_buffer_0)
#pragma ivdep array(feature_win_buffer_1)
#pragma ivdep array(feature_win_buffer_2)
#pragma ivdep array(feature_win_buffer_3)
#pragma ivdep array(feature_win_buffer_4)
#pragma ivdep array(feature_win_buffer_5)
#pragma ivdep array(feature_win_buffer_6)
#pragma ivdep array(feature_win_buffer_7)
#pragma ivdep array(feature_win_buffer_8)
#pragma ivdep array(feature_win_buffer_9)
#pragma ivdep array(feature_win_buffer_10)
#pragma ivdep array(feature_win_buffer_11)
#pragma ivdep array(feature_win_buffer_12)
#if PE_NUM_Y>11
#pragma ivdep array(feature_win_buffer_13)
#pragma ivdep array(feature_win_buffer_14)
#endif
#ifdef DEF_BUF_WIDTH
#pragma ivdep array(weight_buffer_0)
#pragma ivdep array(weight_buffer_1)
#pragma ivdep array(weight_buffer_2)
#else
#pragma ivdep array(weight_buffer)
#endif
Item:
        for(unsigned int  win_itm_xyz=0; win_itm_xyz<item_loop_bound; win_itm_xyz++) {
            //// The following loops are flattened as the upper loop to improve pipeline efficiency
            //for(unsigned short win_itm_z=0; win_itm_z<weight_dim3/VEC_SIZE; win_itm_z++){
            //	for(unsigned char  win_itm_y=0; win_itm_y<weight_dim2*CONV_GP_SIZE_Y; win_itm_y++){
            //		for(unsigned char  win_itm_x=0; win_itm_x<weight_dim1*CONV_GP_SIZE_X; win_itm_x++){

            // Winbuffer loading operations
            if(win_itm_z<(weight_dim3/VEC_SIZE) && out_idx_xyz<conv_group_num_dim1x2_dim4_div_lane) {
            
#ifdef STRIDE_1
		        if(stride==1){
		        	feature_win_buffer_rd_addr = win_itm_z*win_size_x + win_itm_x;
		        }
#endif
#ifdef STRIDE_4
		        else if(stride==4){
		        	feature_win_buffer_rd_addr = win_itm_z*win_size_x*4 + win_itm_y_cycling*win_size_x + win_itm_x;
		        }
#endif

                feature_idx_dim1 = win_itm_x + gp_num_x_winbuf*gp_size_x*stride;
                feature_idx_dim2_port0 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride;
                feature_idx_dim2_port1 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride + PE_NUM_Y_DIV;
                feature_idx_dim2_port2 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride + PE_NUM_Y_DIV*2;
                feature_idx_dim2_port3 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride + PE_NUM_Y_DIV*3;
                feature_idx_dim2_port4 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride + PE_NUM_Y_DIV*4;
                feature_idx_dim3 = win_itm_z;

                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port0>=padding && feature_idx_dim2_port0<data_dim2+padding)) {
                    data_vec_port0 = bottom0[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port0-padding)*data_dim1 + (feature_idx_dim1-padding)];
                } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                            // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                    #pragma unroll
                    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                        data_vec_port0.data[vv] = CZERO;
                    }
                }
                switch(win_itm_y_offset)
                {
                case 0:
                    feature_win_buffer_0[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 1:
                    feature_win_buffer_1[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 2:
                    feature_win_buffer_2[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                // case 3:
                //     feature_win_buffer_3[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                // case 4:
                //     feature_win_buffer_4[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                }

                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port1>=padding && feature_idx_dim2_port1<data_dim2+padding)) {
                    data_vec_port1 = bottom1[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port1-padding)*data_dim1 + (feature_idx_dim1-padding)];
                } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                            // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                    #pragma unroll
                    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                        data_vec_port1.data[vv] = CZERO;
                    }
                }
                switch(win_itm_y_offset)
                {
                case 0:
                    feature_win_buffer_3[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                case 1:
                    feature_win_buffer_4[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                case 2:
                    feature_win_buffer_5[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                // case 3:
                //     feature_win_buffer_8[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                // case 4:
                //     feature_win_buffer_9[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                }

                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port2>=padding && feature_idx_dim2_port2<data_dim2+padding)) {
                    data_vec_port2 = bottom2[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port2-padding)*data_dim1 + (feature_idx_dim1-padding)];
                } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                            // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                    #pragma unroll
                    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                        data_vec_port2.data[vv] = CZERO;
                    }
                }
                // if((win_item_y_port+PE_NUM_Y_DIV*2)<gp_size_y) //for 1x1 conv, don't need padding
                switch(win_itm_y_offset)
                {
                case 0:
                    feature_win_buffer_6[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                case 1:
                    feature_win_buffer_7[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                case 2:
                    feature_win_buffer_8[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                // case 3:
                //     feature_win_buffer_13[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                // case 4:
                //     feature_win_buffer_14[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                }

                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port3>=padding && feature_idx_dim2_port3<data_dim2+padding)) {
                    data_vec_port3 = bottom3[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port3-padding)*data_dim1 + (feature_idx_dim1-padding)];
                } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                            // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                    #pragma unroll
                    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                        data_vec_port3.data[vv] = CZERO;
                    }
                }
                switch(win_itm_y_offset)
                {
                case 0:
                    feature_win_buffer_9[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port3; break;
                case 1:
                    feature_win_buffer_10[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port3; break;
                case 2:
                    feature_win_buffer_11[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port3; break;
                }

                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port4>=padding && feature_idx_dim2_port4<data_dim2+padding)) {
                    data_vec_port4 = bottom4[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port4-padding)*data_dim1 + (feature_idx_dim1-padding)];
                } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                            // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                    #pragma unroll
                    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                        data_vec_port4.data[vv] = CZERO;
                    }
                }
                switch(win_itm_y_offset)
                {
                case 0:
                    feature_win_buffer_12[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port4; break;
#if PE_NUM_Y>11
                case 1:
                    feature_win_buffer_13[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port4; break;
                case 2:
                    feature_win_buffer_14[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port4; break;
#else
                default:
                    break;
#endif
                }

#ifdef DEBUG_MEMRD
                // if(gp_num_x_winbuf==0 && gp_num_y_winbuf==1 && out_idx_z_winbuf==0 && win_itm_y_offset==0 && win_itm_x<10){
                // printf("work-item x=%d, y=%d, z=%d, offset=%d, write data in channel 0=%f\n", win_itm_x, win_itm_y_port, win_itm_z, data_offset, (float)feature_win_buffer_0[(~flag)&0x01][feature_win_buffer_rd_addr].data[0]);
                // printf("work-item x=%d, y=%d, z=%d, write weight in channel 0=%f\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, (float)weight_ch_vec.data[0].lane[0].data[0]);
                // }
                if(gp_num_x_winbuf==0 && gp_num_y_winbuf<=2 && out_idx_z_winbuf==0){
                printf("x=%d\ty=%d\tz=%d\ty_cyc=%d\ty_off=%d\n", win_itm_x, win_itm_y_port, win_itm_z, win_itm_y_cycling, win_itm_y_offset);
                // printf("win_itm_z=%d\tgp_x=%d\tgp_y=%d\tgp_z=%d\tgp_win_x=%d\tgp_win_y=%d\tgp_win_z=%d\n", win_itm_z, gp_num_x, gp_num_y, out_idx_z, gp_num_x_winbuf, gp_num_y_winbuf, out_idx_z_winbuf);
                }
#endif

                // selecting write port
		        if((win_itm_y_port==PE_NUM_Y_DIV-1) && (win_itm_x==win_size_x-1))
		        	win_itm_y_offset = 0;
		        else if((win_itm_x==win_size_x-1) && (win_itm_y_cycling==stride-1))
		        	win_itm_y_offset++;

		        if(((win_itm_y_cycling==stride-1)||(win_itm_y_port==PE_NUM_Y_DIV-1)) && (win_itm_x==win_size_x-1))
		        	win_itm_y_cycling = 0;
		        else if(win_itm_x==win_size_x-1)
		        	win_itm_y_cycling++;

                // used as virtual loop counters
                if((win_itm_y_port==PE_NUM_Y_DIV-1) && (win_itm_x==win_size_x-1)){ 
                    win_itm_y_port = 0;
                    win_itm_z++;
                }else if(win_itm_x==win_size_x-1){
                    win_itm_y_port++;
                }
                if(win_itm_x==win_size_x-1){
                    win_itm_x = 0;
                }else{
                    win_itm_x++;
                }
            }

            // Load weight into weight buffer
            if(gp_item_idx_x==0 && out_idx_xyz>0) {

                weight_ch_tmp = weights[out_idx_z*weight_dim1x2x3/VEC_SIZE + output_idx_dim3*weight_dim1x2 + output_idx_dim2*weight_dim1 + output_idx_dim1];
#ifdef DEF_BUF_WIDTH
                weight_addr = output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1;
                WRITE_BUF(weight_buffer, weight_addr, weight_ch_tmp);
#else                
                weight_buffer[output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1] = weight_ch_tmp;
#endif

            }

            // Only output data for valid convolution work-items
            // In this version, grouping is only performed in row (x) direction
            // if(gp_num_x*gp_size_x+gp_item_idx_x<conv_x && out_idx_xyz>0) {
            if(gp_item_idx_x<win_item && out_idx_xyz>0) {

                if(output_idx_dim1==0 && output_idx_dim2==0 && output_idx_dim3==0) {
                    bias_ch_in = bias[out_idx_z];
                    write_channel_intel(bias_ch, bias_ch_in);

                    // #ifdef DEBUG_MEMRD
                    //printf("work-item x=%d, y=%d, z=%d, channel =0, write bias=%d\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, bias_ch_in.lane[0]);
                    //#endif
                }

                // data
                // feature_y_parallel = win_buffer[flag][output_idx_dim3*win_size_y*win_size_x + output_idx_dim2*win_size_x + (output_idx_dim1+gp_item_idx_x*stride)];

#ifdef STRIDE_1
		        if(stride==1){
		        	idx_y_cycling = 0;
		        	idx_y_offset = output_idx_dim2;
		        	feature_win_buffer_wr_addr = output_idx_dim3*win_size_x + output_idx_dim1+gp_item_idx_x;
		        }
#endif
#ifdef STRIDE_4		
		        else if(stride==4){
		        	idx_y_cycling = output_idx_dim2&0x03;
		        	idx_y_offset = output_idx_dim2>>2;
		        	feature_win_buffer_wr_addr = output_idx_dim3*4*win_size_x + idx_y_cycling*win_size_x + output_idx_dim1+gp_item_idx_x*4;
		        }
#endif	

                feature_y_parallel[0] = feature_win_buffer_0[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[1] = feature_win_buffer_1[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[2] = feature_win_buffer_2[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[3] = feature_win_buffer_3[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[4] = feature_win_buffer_4[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[5] = feature_win_buffer_5[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[6] = feature_win_buffer_6[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[7] = feature_win_buffer_7[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[8] = feature_win_buffer_8[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[9] = feature_win_buffer_9[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[10] = feature_win_buffer_10[flag][feature_win_buffer_wr_addr];
                // for padding when weight_w is 3
                feature_y_parallel[11] = feature_win_buffer_11[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[12] = feature_win_buffer_12[flag][feature_win_buffer_wr_addr];
#if PE_NUM_Y>11
                feature_y_parallel[13] = feature_win_buffer_13[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[14] = feature_win_buffer_14[flag][feature_win_buffer_wr_addr];
#endif


                switch(idx_y_offset)
		        {
                    case 0:
			        #pragma unroll
			        for(uchar yy=0; yy<PE_NUM_Y; yy++){feature_y.data[yy] = feature_y_parallel[0+yy];} break;
			        case 1:
			        #pragma unroll
			        for(uchar yy=0; yy<PE_NUM_Y; yy++){feature_y.data[yy] = feature_y_parallel[1+yy];} break;
			        case 2: // to support all layers (kernel size=3, stride=1)
			        #pragma unroll
			        for(uchar yy=0; yy<PE_NUM_Y; yy++){feature_y.data[yy] = feature_y_parallel[2+yy];} break;
#ifndef YOLO        
                    case 3: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[3]; break;
			        case 4: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[4]; break;
			        case 5: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[5]; break;
			        case 6: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[6]; break;
			        case 7: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[7]; break;
			        case 8: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[8]; break;
			        case 9: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[9]; break;
			        case 10: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[10]; break;
			        case 11: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[11]; break;
			        case 12: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[12]; break;
			        case 13: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[13]; break;
			        case 14: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[14]; break;
			        // case 15: // for FC layer, only read single line
			        // feature_y.data[0] = feature_y_parallel[15]; break;
#endif
                }

#ifdef CHANNEL_OPT
                write_channel_intel(data_ch, feature_y);
#else
                #pragma unroll
                for(unsigned char ll=0; ll<LANE_NUM; ll++) {
                    data_ch_vec.lane[ll] = feature_y;
                }
                write_channel_intel(data_ch, data_ch_vec);
#endif

                // weight and bias fetcher
#ifdef DEF_BUF_WIDTH
                weight_addr = output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1;
                READ_BUF(weight_buffer, weight_addr, weight_ch_tmp);
#else
                weight_ch_tmp = weight_buffer[output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1];
                //weight_ch_tmp = weights[out_idx_z*weight_dim1x2x3/VEC_SIZE + output_idx_dim3*weight_dim1x2 + output_idx_dim2*weight_dim1 + output_idx_dim1];
#endif 
#ifdef CHANNEL_OPT 
                write_channel_intel(weight_ch, weight_ch_tmp);
#else      
                #pragma unroll
				for(unsigned char yy=0; yy<PE_NUM_Y; yy++){
					weight_ch_vec.data[yy] = weight_ch_tmp;
				}
                write_channel_intel(weight_ch, weight_ch_vec);
#endif

#ifdef DEBUG_MEMRD
                // if(gp_num_x==0 && gp_num_y==1 && out_idx_z==0 && gp_item_idx_x==0){
                // printf("work-item x=%d, y=%d, z=%d, offset=%d, write data in channel 0=%f\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, data_offset, (float)data_ch_vec.lane[0].data[0].data[0]);
                // printf("work-item x=%d, y=%d, z=%d, write weight in channel 0=%f\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, (float)weight_ch_vec.data[0].lane[0].data[0]);
                // printf("%d\t", feature_win_buffer_0[flag][feature_win_buffer_wr_addr].data[0]);
                // }

                if(gp_num_y==0 && out_idx_z==0 && output_idx_dim2==0 && output_idx_dim1==0 && output_idx_dim3==0)
                // printf("out_dim1=%d\tout_dim2=%d\tout_dim3=%d\titm_idx=%d\tclcying=%d\toffset=%d\n",
                //         output_idx_dim1, output_idx_dim2, output_idx_dim3, gp_item_idx_x, idx_y_cycling, idx_y_offset);
                printf("gp_num_x=%d\tgp_size_x=%d\tgp_item_idx_x=%d\ttotal=%d\n", gp_num_x, gp_size_x, gp_item_idx_x, gp_num_x*gp_size_x+gp_item_idx_x);
#endif

                // used as output loop counters
                if((output_idx_dim3==weight_dim3/VEC_SIZE-1) && (output_idx_dim2==weight_dim2-1) && (output_idx_dim1==weight_dim1-1)) {
                    output_idx_dim3 = 0;
                    gp_item_idx_x++;
                } else if((output_idx_dim2==weight_dim2-1)&& (output_idx_dim1==weight_dim1-1))
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

        }

        //		}// end of win_itm_z
        //	}// end of win_itm_y
        //}// end of win_itm_x

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

        // used as virtual group loop counters
        if((out_idx_z==weight_dim4_div_lane-1) && (gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
            out_idx_z = 0;
        else if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
            out_idx_z++;

        if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
            gp_num_y = 0;
        else if(gp_num_x==group_num_x-1)
            gp_num_y++;

        if(gp_num_x==group_num_x-1)
            gp_num_x = 0;
        else
            gp_num_x++;

        // if(out_idx_z>=6)
        // printf("gp_x=%d\tgp_y=%d\tgp_z=%d\tgp_win_x=%d\tgp_win_y=%d\tgp_win_z=%d\n", gp_num_x, gp_num_y, out_idx_z, gp_num_x_winbuf, gp_num_y_winbuf, out_idx_z_winbuf);


        //			}// end of gp_num_x
        //		}// end of gp_num_y
        //}// end of out_idx_z
    }

    //printf("Kernel 0 lanched !!!\n");
}





#elif defined RD_MULTPORT_5_4  // for PE_NUM_Y = 15 / 17
// Fetch Data from Global Memory
__kernel
__attribute__((task))
__attribute__((max_global_work_dim(0)))
void memRead(
    // Params Ports
    ushort  data_dim1,
    ushort  data_dim2,
    uint data_dim1xdim2,
    uchar  weight_dim1,
    uchar  weight_dim2,
    ushort weight_dim3,
    ushort weight_dim4_div_lane, // avoid generating divider
    uchar  weight_dim1x2,
    uint   weight_dim1x2x3,
    ushort  conv_x,
    //uchar  conv_y,           // not used in this version
    uchar  stride,
    uchar  padding,
    uchar  split,
    uchar gp_size_y,
    ushort gp_last_size_x,
    ushort gp_size_x,
    ushort  group_num_x,
    ushort  group_num_y,
    uint  conv_group_num_dim1x2_dim4_div_lane,
    // uchar  group_rem_size_x,
    //uchar  group_rem_size_y, // not used in this version
    // uint   group_rem_size_xyz,
    ushort win_size_x,
    // uchar  win_size_y_port,
    uint   win_size_xyz_port,
    uint Item_Loop_Bound,
    uint Item_Last_Loop_Bound,
    // Data Ports
    __global DPTYPE_VEC    *restrict bottom0,
    __global DPTYPE_VEC    *restrict bottom1,
    __global DPTYPE_VEC    *restrict bottom2,
    __global DPTYPE_VEC    *restrict bottom3,
    __global DPTYPE_VEC    *restrict bottom4,
    __global DPTYPE_SCAL_VEC  *restrict weights,
    __global volatile DPTYPE_SCAL *restrict bias        
){

    // feature win buffer
    // Ping-pong buffer
    __local DPTYPE_VEC  feature_win_buffer_0[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_1[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_2[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_3[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_4[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_5[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_6[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_7[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_8[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_9[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_10[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_11[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_12[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_13[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_14[2][WIN_BUF_DEPTH];
    // for padding when weight_w is 3
    __local DPTYPE_VEC  feature_win_buffer_15[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_16[2][WIN_BUF_DEPTH];
#if PE_NUM_Y>15
    __local DPTYPE_VEC  feature_win_buffer_17[2][WIN_BUF_DEPTH];
    __local DPTYPE_VEC  feature_win_buffer_18[2][WIN_BUF_DEPTH];
#endif

    // Weight buffer
#ifdef DEF_BUF_WIDTH
    __local DPTYPE_VEC DEFINE_BUF(weight_buffer, WEIGHT_BUF_SIZE);
#else
    __local DPTYPE_SCAL_VEC  weight_buffer[WEIGHT_BUF_SIZE];
#endif

    // Input Data, Weights and Bias
    // read from feature_win_buffer_0 (1,2,3,...,14)
    DPTYPE_VEC    feature_y_parallel[PE_NUM_Y+WIN_BUF_Y_PAD];
    // take data of PE_NUM_Y from feature_y_parallel
    // DPTYPE_PE_VEC    feature_y;
    // read from bottom
    DPTYPE_VEC    data_vec_port0;
    DPTYPE_VEC    data_vec_port1;
    DPTYPE_VEC    data_vec_port2;
    DPTYPE_VEC    data_vec_port3;
    DPTYPE_VEC    data_vec_port4;
    // write to kl_conv
    SCAL_PE_VEC   data_ch_vec;
    DPTYPE_SCAL_VEC weight_ch_tmp;
    PE_SCAL_VEC   weight_ch_vec;
    DPTYPE_SCAL  bias_ch_in;

    // feature win buffer read/write address
    // ushort     feature_win_buffer_rd_addr, feature_win_buffer_wr_addr;
    ushort     data_offset = 0; // assuming the 1st layer is not in split
    uchar  flag; // ping-pong flag

    // virtual loop counters
    // for loading feature to feature_win_buffer from bottom
    ushort win_itm_z;

	uchar  win_itm_y_port;
	// uchar  win_itm_y_cycling, win_itm_y_offset;
    // for transfer data to knl_conv from feature_win_buffer
    ushort output_idx_dim3;
    uchar  output_idx_dim1, output_idx_dim2;
    // uchar  idx_y_cycling, idx_y_offset;
    // for group counters
    short gp_num_x;
    ushort gp_num_y, out_idx_z;
    ushort gp_num_x_winbuf, gp_num_y_winbuf, out_idx_z_winbuf;

    // for global index
    ushort feature_idx_dim1;  // for yolov2, maximum is 418(padding)
    ushort feature_idx_dim2_port0; // for yolov2, maximum is 418(padding)
    ushort feature_idx_dim2_port1;
    ushort feature_idx_dim2_port2;
    ushort feature_idx_dim2_port3;
    ushort feature_idx_dim2_port4;
    ushort feature_idx_dim3;
    // counter the number of x-dimensional convolutions in the feature_win_buffer
    ushort  gp_item_idx_x;
    ushort  win_itm_x;
    ushort  win_item;
    uint   item_loop_bound;

    gp_num_x_winbuf = 0; // there is only one group for FC mode when batch=1
    gp_num_y_winbuf = 0;
    out_idx_z_winbuf = 0;

    // reset global group virtual loop counters
    gp_num_x = -1;
    gp_num_y = 0;
    out_idx_z = 0;

#ifdef DEF_BUF_WIDTH
#pragma ivdep array(weight_buffer_0)
#pragma ivdep array(weight_buffer_1)
#pragma ivdep array(weight_buffer_2)
#else
#pragma ivdep array(weight_buffer)
#endif        
Group:
    //printf("%d\n", weight_dim4_div_lane*group_num_y*group_num_x);
    //printf("%d\t%d\t%d\n", weight_dim4_div_lane, group_num_y, group_num_x);
    // printf("memRd: %d\n", conv_group_num_dim1x2_dim4_div_lane);

    for(unsigned int out_idx_xyz=0; out_idx_xyz<=conv_group_num_dim1x2_dim4_div_lane; out_idx_xyz++) {
        // The following group loops are flattened as the upper loop to improve pipeline efficiency
        //for(unsigned short out_idx_z=0; out_idx_z<weight_dim4_div_lane; out_idx_z++){
        
        // special case when split==1, the output feature maps depend on only half the input feature maps
        if(split==0)
            data_offset = 0;
        else if(out_idx_z_winbuf<(weight_dim4_div_lane>>1)) // the lower half of the output feature maps depend on the lower half of the input
            data_offset = 0;
        else
            data_offset = weight_dim3/VEC_SIZE;	// the upper half of the output feature maps depend on the upper half of the input

        //for(unsigned short gp_num_y=0; gp_num_y<group_num_y; gp_num_y++){
        //for(unsigned short gp_num_x=0; gp_num_x<group_num_x+1; gp_num_x++){ // add one more extra iteration for ping-pong buffering operations

        flag = out_idx_xyz & 0x01; //ping-pong flag
        
        // feature win buffer read/write address
        ushort     feature_win_buffer_rd_addr, feature_win_buffer_wr_addr;
#ifdef DEF_BUF_WIDTH
        ushort     weight_addr;
#endif
        uchar  win_itm_y_cycling, win_itm_y_offset;
        uchar  idx_y_cycling, idx_y_offset;
        // PE_SCAL_VEC    feature_y;
        DPTYPE_PE_VEC feature_y;

        // reset output loop counters
        output_idx_dim1 = 0;
        output_idx_dim2 = 0;
        output_idx_dim3 = 0;
        // reset in-group item counters
        gp_item_idx_x = 0;
        
        win_itm_y_offset = 0;

        // reset input winbuffer loop counters
        win_itm_x = 0;
        win_itm_y_port = 0;
        win_itm_z = 0;


        if(gp_num_x==group_num_x-1) {// last group in each row
            // ensuring that both winbuf load loop and output loop are finished, i.e., use a larger value as the loop bound
            // item_loop_bound = win_size_x>=group_rem_size_x?(win_size_xyz/VEC_SIZE):(group_rem_size_xyz/VEC_SIZE);
            item_loop_bound = Item_Last_Loop_Bound;
            win_item = gp_last_size_x;
        }else {
            // if(stride>=weight_dim1 || stride>=weight_dim2) // special case convolution layers with stride>weight_dim1/2, such as resnet50
            //     item_loop_bound = win_size_xyz/VEC_SIZE;
            // else
                // item_loop_bound = (weight_dim1x2x3*CONV_GP_SIZE_Y*CONV_GP_SIZE_X/VEC_SIZE);
                item_loop_bound = Item_Loop_Bound;
                win_item = gp_size_x;
        }

#pragma ivdep array(feature_win_buffer_0)
#pragma ivdep array(feature_win_buffer_1)
#pragma ivdep array(feature_win_buffer_2)
#pragma ivdep array(feature_win_buffer_3)
#pragma ivdep array(feature_win_buffer_4)
#pragma ivdep array(feature_win_buffer_5)
#pragma ivdep array(feature_win_buffer_6)
#pragma ivdep array(feature_win_buffer_7)
#pragma ivdep array(feature_win_buffer_8)
#pragma ivdep array(feature_win_buffer_9)
#pragma ivdep array(feature_win_buffer_10)
#pragma ivdep array(feature_win_buffer_11)
#pragma ivdep array(feature_win_buffer_12)
#pragma ivdep array(feature_win_buffer_13)
#pragma ivdep array(feature_win_buffer_14)
#pragma ivdep array(feature_win_buffer_15)
#pragma ivdep array(feature_win_buffer_16)
#if PE_NUM_Y>15
#pragma ivdep array(feature_win_buffer_17)
#pragma ivdep array(feature_win_buffer_18)
#endif
#ifdef DEF_BUF_WIDTH
#pragma ivdep array(weight_buffer_0)
#pragma ivdep array(weight_buffer_1)
#pragma ivdep array(weight_buffer_2)
#else
#pragma ivdep array(weight_buffer)
#endif
Item:
        for(unsigned int  win_itm_xyz=0; win_itm_xyz<item_loop_bound; win_itm_xyz++) {
            //// The following loops are flattened as the upper loop to improve pipeline efficiency
            //for(unsigned short win_itm_z=0; win_itm_z<weight_dim3/VEC_SIZE; win_itm_z++){
            //	for(unsigned char  win_itm_y=0; win_itm_y<weight_dim2*CONV_GP_SIZE_Y; win_itm_y++){
            //		for(unsigned char  win_itm_x=0; win_itm_x<weight_dim1*CONV_GP_SIZE_X; win_itm_x++){

            // Winbuffer loading operations
            if(win_itm_z<(weight_dim3/VEC_SIZE) && out_idx_xyz<conv_group_num_dim1x2_dim4_div_lane) {
            
#ifdef STRIDE_1
		        if(stride==1){
		        	feature_win_buffer_rd_addr = win_itm_z*win_size_x + win_itm_x;
		        }
#endif
#ifdef STRIDE_4
		        else if(stride==4){
		        	feature_win_buffer_rd_addr = win_itm_z*win_size_x*4 + win_itm_y_cycling*win_size_x + win_itm_x;
		        }
#endif

                feature_idx_dim1 = win_itm_x + gp_num_x_winbuf*gp_size_x*stride;
                feature_idx_dim2_port0 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride;
                feature_idx_dim2_port1 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride + PE_NUM_Y_DIV;
                feature_idx_dim2_port2 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride + PE_NUM_Y_DIV*2;
                feature_idx_dim2_port3 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride + PE_NUM_Y_DIV*3;
                feature_idx_dim2_port4 = win_itm_y_port + gp_num_y_winbuf*gp_size_y*stride + PE_NUM_Y_DIV*4;
                feature_idx_dim3 = win_itm_z;

                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port0>=padding && feature_idx_dim2_port0<data_dim2+padding)) {
                    data_vec_port0 = bottom0[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port0-padding)*data_dim1 + (feature_idx_dim1-padding)];
                } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                            // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                    #pragma unroll
                    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                        data_vec_port0.data[vv] = CZERO;
                    }
                }
                switch(win_itm_y_offset)
                {
                case 0:
                    feature_win_buffer_0[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 1:
                    feature_win_buffer_1[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 2:
                    feature_win_buffer_2[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                case 3:
                    feature_win_buffer_3[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                // case 4:
                //     feature_win_buffer_4[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port0; break;
                }

                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port1>=padding && feature_idx_dim2_port1<data_dim2+padding)) {
                    data_vec_port1 = bottom1[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port1-padding)*data_dim1 + (feature_idx_dim1-padding)];
                } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                            // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                    #pragma unroll
                    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                        data_vec_port1.data[vv] = CZERO;
                    }
                }
                switch(win_itm_y_offset)
                {
                case 0:
                    feature_win_buffer_4[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                case 1:
                    feature_win_buffer_5[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                case 2:
                    feature_win_buffer_6[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                case 3:
                    feature_win_buffer_7[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                // case 4:
                //     feature_win_buffer_9[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port1; break;
                }

                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port2>=padding && feature_idx_dim2_port2<data_dim2+padding)) {
                    data_vec_port2 = bottom2[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port2-padding)*data_dim1 + (feature_idx_dim1-padding)];
                } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                            // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                    #pragma unroll
                    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                        data_vec_port2.data[vv] = CZERO;
                    }
                }
                // if((win_item_y_port+PE_NUM_Y_DIV*2)<gp_size_y) //for 1x1 conv, don't need padding
                switch(win_itm_y_offset)
                {
                case 0:
                    feature_win_buffer_8[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                case 1:
                    feature_win_buffer_9[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                case 2:
                    feature_win_buffer_10[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                case 3:
                    feature_win_buffer_11[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                // case 4:
                //     feature_win_buffer_14[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port2; break;
                }

                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port3>=padding && feature_idx_dim2_port3<data_dim2+padding)) {
                    data_vec_port3 = bottom3[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port3-padding)*data_dim1 + (feature_idx_dim1-padding)];
                } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                            // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                    #pragma unroll
                    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                        data_vec_port3.data[vv] = CZERO;
                    }
                }
                switch(win_itm_y_offset)
                {
                case 0:
                    feature_win_buffer_12[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port3; break;
                case 1:
                    feature_win_buffer_13[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port3; break;
                case 2:
                    feature_win_buffer_14[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port3; break;
                case 3:
                    feature_win_buffer_15[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port3; break;
                }

                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2_port4>=padding && feature_idx_dim2_port4<data_dim2+padding)) {
                    data_vec_port4 = bottom4[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2_port4-padding)*data_dim1 + (feature_idx_dim1-padding)];
                } else { // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
                            // or invalid work-item in the last group set feature map to zeros (feature_idx>=data_dim+2*padding)
                    #pragma unroll
                    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                        data_vec_port4.data[vv] = CZERO;
                    }
                }
                switch(win_itm_y_offset)
                {
                case 0:
                    feature_win_buffer_16[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port4; break;
#if PE_NUM_Y>15
                case 1:
                    feature_win_buffer_17[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port4; break;
                case 2:
                    feature_win_buffer_18[(~flag)&0x01][feature_win_buffer_rd_addr] = data_vec_port4; break;
#endif
                default:
                    break;
                }

#ifdef DEBUG_MEMRD
                // if(gp_num_x_winbuf==0 && gp_num_y_winbuf==1 && out_idx_z_winbuf==0 && win_itm_y_offset==0 && win_itm_x<10){
                // printf("work-item x=%d, y=%d, z=%d, offset=%d, write data in channel 0=%f\n", win_itm_x, win_itm_y_port, win_itm_z, data_offset, (float)feature_win_buffer_0[(~flag)&0x01][feature_win_buffer_rd_addr].data[0]);
                // printf("work-item x=%d, y=%d, z=%d, write weight in channel 0=%f\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, (float)weight_ch_vec.data[0].lane[0].data[0]);
                // }
                if(gp_num_x_winbuf==0 && gp_num_y_winbuf<=2 && out_idx_z_winbuf==0){
                printf("x=%d\ty=%d\tz=%d\ty_cyc=%d\ty_off=%d\n", win_itm_x, win_itm_y_port, win_itm_z, win_itm_y_cycling, win_itm_y_offset);
                // printf("win_itm_z=%d\tgp_x=%d\tgp_y=%d\tgp_z=%d\tgp_win_x=%d\tgp_win_y=%d\tgp_win_z=%d\n", win_itm_z, gp_num_x, gp_num_y, out_idx_z, gp_num_x_winbuf, gp_num_y_winbuf, out_idx_z_winbuf);
                }
#endif

                // selecting write port
		        if((win_itm_y_port==PE_NUM_Y_DIV-1) && (win_itm_x==win_size_x-1))
		        	win_itm_y_offset = 0;
		        else if((win_itm_x==win_size_x-1) && (win_itm_y_cycling==stride-1))
		        	win_itm_y_offset++;

		        if(((win_itm_y_cycling==stride-1)||(win_itm_y_port==PE_NUM_Y_DIV-1)) && (win_itm_x==win_size_x-1))
		        	win_itm_y_cycling = 0;
		        else if(win_itm_x==win_size_x-1)
		        	win_itm_y_cycling++;

                // used as virtual loop counters
                if((win_itm_y_port==PE_NUM_Y_DIV-1) && (win_itm_x==win_size_x-1)){ 
                    win_itm_y_port = 0;
                    win_itm_z++;
                }else if(win_itm_x==win_size_x-1){
                    win_itm_y_port++;
                }
                if(win_itm_x==win_size_x-1){
                    win_itm_x = 0;
                }else{
                    win_itm_x++;
                }
            }

            // Load weight into weight buffer
            if(gp_item_idx_x==0 && out_idx_xyz>0) {

                weight_ch_tmp = weights[out_idx_z*weight_dim1x2x3/VEC_SIZE + output_idx_dim3*weight_dim1x2 + output_idx_dim2*weight_dim1 + output_idx_dim1];
#ifdef DEF_BUF_WIDTH
                weight_addr = output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1;
                WRITE_BUF(weight_buffer, weight_addr, weight_ch_tmp);
#else                
                weight_buffer[output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1] = weight_ch_tmp;
#endif

            }

            // Only output data for valid convolution work-items
            // In this version, grouping is only performed in row (x) direction
            // if(gp_num_x*gp_size_x+gp_item_idx_x<conv_x && out_idx_xyz>0) {
            if(gp_item_idx_x<win_item && out_idx_xyz>0) {

                if(output_idx_dim1==0 && output_idx_dim2==0 && output_idx_dim3==0) {
                    bias_ch_in = bias[out_idx_z];
                    write_channel_intel(bias_ch, bias_ch_in);

                    // #ifdef DEBUG_MEMRD
                    //printf("work-item x=%d, y=%d, z=%d, channel =0, write bias=%d\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, bias_ch_in.lane[0]);
                    //#endif
                }

                // data
                // feature_y_parallel = win_buffer[flag][output_idx_dim3*win_size_y*win_size_x + output_idx_dim2*win_size_x + (output_idx_dim1+gp_item_idx_x*stride)];

#ifdef STRIDE_1
		        if(stride==1){
		        	idx_y_cycling = 0;
		        	idx_y_offset = output_idx_dim2;
		        	feature_win_buffer_wr_addr = output_idx_dim3*win_size_x + output_idx_dim1+gp_item_idx_x;
		        }
#endif
#ifdef STRIDE_4		
		        else if(stride==4){
		        	idx_y_cycling = output_idx_dim2&0x03;
		        	idx_y_offset = output_idx_dim2>>2;
		        	feature_win_buffer_wr_addr = output_idx_dim3*4*win_size_x + idx_y_cycling*win_size_x + output_idx_dim1+gp_item_idx_x*4;
		        }
#endif	

                feature_y_parallel[0] = feature_win_buffer_0[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[1] = feature_win_buffer_1[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[2] = feature_win_buffer_2[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[3] = feature_win_buffer_3[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[4] = feature_win_buffer_4[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[5] = feature_win_buffer_5[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[6] = feature_win_buffer_6[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[7] = feature_win_buffer_7[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[8] = feature_win_buffer_8[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[9] = feature_win_buffer_9[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[10] = feature_win_buffer_10[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[11] = feature_win_buffer_11[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[12] = feature_win_buffer_12[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[13] = feature_win_buffer_13[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[14] = feature_win_buffer_14[flag][feature_win_buffer_wr_addr];
                // for padding when weight_w is 3
                feature_y_parallel[15] = feature_win_buffer_15[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[16] = feature_win_buffer_16[flag][feature_win_buffer_wr_addr];
#if PE_NUM_Y>15
                feature_y_parallel[17] = feature_win_buffer_17[flag][feature_win_buffer_wr_addr];
                feature_y_parallel[18] = feature_win_buffer_18[flag][feature_win_buffer_wr_addr];
#endif


                switch(idx_y_offset)
		        {
                    case 0:
			        #pragma unroll
			        for(uchar yy=0; yy<PE_NUM_Y; yy++){feature_y.data[yy] = feature_y_parallel[0+yy];} break;
			        case 1:
			        #pragma unroll
			        for(uchar yy=0; yy<PE_NUM_Y; yy++){feature_y.data[yy] = feature_y_parallel[1+yy];} break;
			        case 2: // to support all layers (kernel size=3, stride=1)
			        #pragma unroll
			        for(uchar yy=0; yy<PE_NUM_Y; yy++){feature_y.data[yy] = feature_y_parallel[2+yy];} break;
#ifndef YOLO        
                    case 3: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[3]; break;
			        case 4: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[4]; break;
			        case 5: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[5]; break;
			        case 6: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[6]; break;
			        case 7: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[7]; break;
			        case 8: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[8]; break;
			        case 9: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[9]; break;
			        case 10: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[10]; break;
			        case 11: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[11]; break;
			        case 12: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[12]; break;
			        case 13: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[13]; break;
			        case 14: // for FC layer, only read single line
			        feature_y.data[0] = feature_y_parallel[14]; break;
			        // case 15: // for FC layer, only read single line
			        // feature_y.data[0] = feature_y_parallel[15]; break;
#endif
                }

#ifdef CHANNEL_OPT
                write_channel_intel(data_ch, feature_y);
#else
                #pragma unroll
                for(unsigned char ll=0; ll<LANE_NUM; ll++) {
                    data_ch_vec.lane[ll] = feature_y;
                }
                write_channel_intel(data_ch, data_ch_vec);
#endif

                // weight and bias fetcher
#ifdef DEF_BUF_WIDTH
                weight_addr = output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1;
                READ_BUF(weight_buffer, weight_addr, weight_ch_tmp);
#else
                weight_ch_tmp = weight_buffer[output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1];
                //weight_ch_tmp = weights[out_idx_z*weight_dim1x2x3/VEC_SIZE + output_idx_dim3*weight_dim1x2 + output_idx_dim2*weight_dim1 + output_idx_dim1];
#endif 
#ifdef CHANNEL_OPT 
                write_channel_intel(weight_ch, weight_ch_tmp);
#else      
                #pragma unroll
				for(unsigned char yy=0; yy<PE_NUM_Y; yy++){
					weight_ch_vec.data[yy] = weight_ch_tmp;
				}
                write_channel_intel(weight_ch, weight_ch_vec);
#endif

#ifdef DEBUG_MEMRD
                // if(gp_num_x==0 && gp_num_y==1 && out_idx_z==0 && gp_item_idx_x==0){
                // printf("work-item x=%d, y=%d, z=%d, offset=%d, write data in channel 0=%f\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, data_offset, (float)data_ch_vec.lane[0].data[0].data[0]);
                // printf("work-item x=%d, y=%d, z=%d, write weight in channel 0=%f\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, (float)weight_ch_vec.data[0].lane[0].data[0]);
                // printf("%d\t", feature_win_buffer_0[flag][feature_win_buffer_wr_addr].data[0]);
                // }

                if(gp_num_y==0 && out_idx_z==0 && output_idx_dim2==0 && output_idx_dim1==0 && output_idx_dim3==0)
                // printf("out_dim1=%d\tout_dim2=%d\tout_dim3=%d\titm_idx=%d\tclcying=%d\toffset=%d\n",
                //         output_idx_dim1, output_idx_dim2, output_idx_dim3, gp_item_idx_x, idx_y_cycling, idx_y_offset);
                printf("gp_num_x=%d\tgp_size_x=%d\tgp_item_idx_x=%d\ttotal=%d\n", gp_num_x, gp_size_x, gp_item_idx_x, gp_num_x*gp_size_x+gp_item_idx_x);
#endif

                // used as output loop counters
                if((output_idx_dim3==weight_dim3/VEC_SIZE-1) && (output_idx_dim2==weight_dim2-1) && (output_idx_dim1==weight_dim1-1)) {
                    output_idx_dim3 = 0;
                    gp_item_idx_x++;
                } else if((output_idx_dim2==weight_dim2-1)&& (output_idx_dim1==weight_dim1-1))
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

        }

        //		}// end of win_itm_z
        //	}// end of win_itm_y
        //}// end of win_itm_x

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

        // used as virtual group loop counters
        if((out_idx_z==weight_dim4_div_lane-1) && (gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
            out_idx_z = 0;
        else if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
            out_idx_z++;

        if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
            gp_num_y = 0;
        else if(gp_num_x==group_num_x-1)
            gp_num_y++;

        if(gp_num_x==group_num_x-1)
            gp_num_x = 0;
        else
            gp_num_x++;

        // if(out_idx_z>=6)
        // printf("gp_x=%d\tgp_y=%d\tgp_z=%d\tgp_win_x=%d\tgp_win_y=%d\tgp_win_z=%d\n", gp_num_x, gp_num_y, out_idx_z, gp_num_x_winbuf, gp_num_y_winbuf, out_idx_z_winbuf);


        //			}// end of gp_num_x
        //		}// end of gp_num_y
        //}// end of out_idx_z
    }

    //printf("Kernel 0 lanched !!!\n");
}

#endif


__kernel
__attribute__((task))
__attribute__((max_global_work_dim(0)))
void coreConv(
    // Params Ports
    uint  output_num,
    uint  conv_loop_cnt,
    uint  contol, //[0]-> relu for yolo and resnet, [0]->bn.[1]->wr(fc) for others
    char  frac_w, // max(conv_weight+bn_weight,bn_bias) for leaky relu // conv_weight for relu
    char  frac_b, //  conv_bias
    char  frac_din, // conv input
    char  frac_dout, // conv output
    char  frac_relu  // frac_dout-bn_dout
) {
#ifndef CHANNEL_OPT
    SCAL_PE_VEC mac_data;
    PE_SCAL_VEC mac_weight;
#endif
    DPTYPE_SCAL bias_ch_out;
    DPTYPE_PE_SCAL conv_ch_in;
    DPTYPE  bias[LANE_NUM];
#ifdef ARBI_PRECISION
    ap_int<SUM_BIT>  conv_out[LANE_NUM][PE_NUM_Y];
    ap_int<SUM_BIT> lane_accum[LANE_NUM][PE_NUM_Y];
    ap_int<SUM_BIT> accum_piped[LANE_NUM][PE_NUM_Y][PIPE_DEPTH];
    ap_int<SUM_BIT> conv_sign_exten[LANE_NUM][PE_NUM_Y];
    ap_int<SUM_BIT> conv_with_rnd_bit[LANE_NUM][PE_NUM_Y];
    ap_int<9> conv_sum_bias[LANE_NUM][PE_NUM_Y];
#else
    MACTYPE conv_out[LANE_NUM][PE_NUM_Y];
    MACTYPE lane_accum[LANE_NUM][PE_NUM_Y];
    MACTYPE accum_piped[LANE_NUM][PE_NUM_Y][PIPE_DEPTH];
    MACTYPE conv_sign_exten[LANE_NUM][PE_NUM_Y];
    MACTYPE conv_with_rnd_bit[LANE_NUM][PE_NUM_Y];
    MACTYPE conv_sum_bias[LANE_NUM][PE_NUM_Y];
#endif
    DPTYPE  conv_final[LANE_NUM][PE_NUM_Y];
    DPTYPE  sign_exten[LANE_NUM][PE_NUM_Y]; 
#ifdef CHANNEL_OPT
    DPTYPE_PE_VEC    mac_data_tmp;
    DPTYPE_SCAL_VEC   mac_weight_tmp;                                      
#endif

    //int counter = 0;

    // each iteration generates one output
    for(unsigned int k=0; k<output_num; k++) {

        bias_ch_out = read_channel_intel(bias_ch);
#ifdef CHANNEL_OPT
        SCAL_PE_VEC mac_data;
        PE_SCAL_VEC mac_weight;
#endif
        #pragma unroll
        for(unsigned char ll=0; ll<LANE_NUM; ll++){
            #pragma unroll
            for(unsigned char yy=0; yy<PE_NUM_Y; yy++) {
            
                conv_out[ll][yy] = CZERO;
                bias[ll] = bias_ch_out.lane[ll]; // pass to reg, avoid compile error
                // initialize the deep pipelined registers which store PIPE_DEPTH copys of partial results
                #pragma unroll
                for(unsigned int p=0; p<PIPE_DEPTH; p++) {
                    accum_piped[ll][yy][p] = MASK_ACCUM & CZERO;
                }
            }
        }
        

        for(int j=0; j<conv_loop_cnt; j++) {

#ifdef CHANNEL_OPT
            mac_data_tmp = read_channel_intel(data_ch);
#ifdef EXTENTED_DIM
            #pragma unroll
            for(unsigned char ll=0; ll<LANE_NUM; ll++) {
                mac_data.lane[ll] = mac_data_tmp;
            }
#endif
            mac_weight_tmp = read_channel_intel(weight_ch);
#ifdef EXTENTED_DIM
            #pragma unroll
			for(unsigned char yy=0; yy<PE_NUM_Y; yy++){
				mac_weight.data[yy] = mac_weight_tmp;
			}
#endif
#else
            // load data and weights for each lane
            mac_data = read_channel_intel(data_ch);
            mac_weight = read_channel_intel(weight_ch);
#endif
            // add results from all lanes
            // accumulate with the last copy
            #pragma unroll
            for(unsigned char ll=0; ll<LANE_NUM; ll++){
                #pragma unroll
                for(unsigned char yy=0; yy<PE_NUM_Y; yy++) {

                    // if(ll==0 && k==0)
                    //     for(char vv=0; vv<VEC_SIZE; vv++)
                    //         printf("dot_cnt=%d data=%d \tweight=%d (vv=%d  yy=%d)\n", k, mac_data.lane[ll].data[yy].data[vv], mac_weight.data[yy].lane[ll].data[vv], vv, yy);
#ifdef EXTENTED_DIM
                    lane_accum[ll][yy] = (MASK_ACCUM & accum_piped[ll][yy][PIPE_DEPTH-1]) + (MASK_MULT & mac(mac_data.lane[ll].data[yy], mac_weight.data[yy].lane[ll]));
#else
                    lane_accum[ll][yy] = (MASK_ACCUM & accum_piped[ll][yy][PIPE_DEPTH-1]) + (MASK_MULT & mac(mac_data_tmp.data[yy], mac_weight_tmp.lane[ll]));
#endif
                    // Shift the pipelined registers backwards
                    #pragma unroll
                    for(unsigned int p=PIPE_DEPTH-1; p>0; p-- ) {
                        accum_piped[ll][yy][p] = MASK_ACCUM & accum_piped[ll][yy][p-1];
                    }

                    // update the first copy
                    accum_piped[ll][yy][0] = MASK_ACCUM & lane_accum[ll][yy];

#ifdef DEBUG_CONV
                    if(k==288*14+1 && yy==9 && j==0){
                    	printf("dot_cnt=%d data=%f weight=%f (loop=%d, lane= %d, vec=0)\n", k, (float)mac_data.lane[ll].data[0].data[0], (float)mac_weight.data[0].lane[ll].data[0], j, ll);
                    }
#endif
                }
            }
        }// end of conv loop

        #pragma unroll
        for(unsigned char yy=0; yy<PE_NUM_Y; yy++){
        // for(unsigned char ll=0; ll<LANE_NUM; ll++) {
            #pragma unroll
             for(unsigned char ll=0; ll<LANE_NUM; ll++) {
            // for(unsigned char yy=0; yy<PE_NUM_Y; yy++){
                // accumulate all the partial results
                #pragma unroll
                for(unsigned i=0; i<PIPE_DEPTH; i++) {
                    conv_out[ll][yy] += accum_piped[ll][yy][i];
                }

                // if(k==288*14+1 && yy==9){
                //     printf("dot_cnt=%d output=%d (yy=%d, lane= %d, vec=0)\n", k, conv_out[ll][yy], yy, ll);
                // }

                // int tmp_conv_out = conv_out[ll][yy];
                // tmp_conv_out = tmp_conv_out >> 20;
                // if(tmp_conv_out==0xffffffff || tmp_conv_out==0){
                //     counter = counter + 1;
                // }else{
                //     printf("\nconv not match: %x  %x\n", conv_out[ll][yy], tmp_conv_out);
                // }
                
                // round and truncate the results to the output precision
                // note: ((frac_w+frac_din)-frac_dout)) should be checked by host to be a positive number
                if(conv_out[ll][yy]>=0){
                    conv_sign_exten[ll][yy] = 0x00;
                    // conv_with_rnd_bit[ll][yy] = (conv_sign_exten[ll][yy] | (conv_out[ll][yy]>>(frac_w+frac_din-frac_dout-1))) + 0x01;
                }else{
                    conv_sign_exten[ll][yy] = ~(0xFFFFFFFF>>(frac_w+frac_din-frac_dout-1)); // ">>" is logic shift, then perform sign extension manually
                    // conv_with_rnd_bit[ll][yy] = (conv_sign_exten[ll][yy] | (conv_out[ll][yy]>>(frac_w+frac_din-frac_dout-1))) + 0x01;
                }
                // First, perform sign extension and the 1st-step rounding before sum with bias
                conv_with_rnd_bit[ll][yy] = (conv_sign_exten[ll][yy] | (conv_out[ll][yy]>>(frac_w+frac_din-frac_dout-1))) + 0x01;

                // Second, deal with Overflow and Underflow cases and the 2nd rounding after sum with bias
                if(conv_with_rnd_bit[ll][yy]>=256)
                    conv_sum_bias[ll][yy] = MASK9B & 0xFF; //=255
                else if(conv_with_rnd_bit[ll][yy]<-256)
                    conv_sum_bias[ll][yy] = MASK9B & 0x100; //=-256
                else
                    conv_sum_bias[ll][yy] = (MASK9B & conv_with_rnd_bit[ll][yy])+(bias[ll]>>(frac_b-frac_dout-1))+0x01;

                // // final truncation
                conv_final[ll][yy] = MASK8B & (conv_sum_bias[ll][yy]>>0x01);  // remove the last rounding bit

                #ifdef DEBUG_CONV
                if(k==288*14+1 && yy==9)
                    printf("dot_cnt=%d yy=%d ll=%d   sum_bias=%x\n", k, yy, ll, conv_final[ll][yy]);
                #endif
                
#if defined BN_FLOAT || defined RELU_FP
                conv_ch_in.data[yy].lane[ll] = conv_final[ll][yy];
            } 
        }
        //BatchNorm
        if(contol==0)
            write_channel_intel(conv_ch, conv_ch_in);
        else//for fc layer no bn,Write
            write_channel_intel(bypass_bn_ch, conv_ch_in);
#else
                // Relu operation
                if((contol&0x01)==0x01) {
                    if((conv_final[ll][yy]&MASKSIGN)==MASKSIGN){ // MSB is sign bit
#if defined LEAKY_0
                        // relu
                        conv_ch_in.data[yy].lane[ll] = 0;
#elif defined LEAKY_0125
                        // leaky relu(0.125) rounding model: to nearst towards infinity 
                        sign_exten[ll][yy] = ~(0xFF>>(frac_relu+3));
                        conv_ch_in.data[yy].lane[ll] = (conv_final[ll][yy] + relu_add_coef[frac_relu+3]) >> (frac_relu+3);
                        if(conv_ch_in.data[yy].lane[ll]!=0x00){
                            conv_ch_in.data[yy].lane[ll] = sign_exten[ll][yy] | conv_ch_in.data[yy].lane[ll];
                        }
#endif
                        // #ifdef DEBUG_CONV
                        // if(yy==0 && k==(52*16*0+52*0+4))
                        //     printf("dot_cnt=%d yy=%d ll=%d   relu=%f (%x, %d)   sum_bias=%x (%d) \n", k, yy, ll, (float)(conv_ch_in.data[yy].lane[ll])*0.125, conv_ch_in.data[yy].lane[ll], conv_ch_in.data[yy].lane[ll], conv_final[ll][yy], conv_final[ll][yy]);
                        // #endif
                    }else{
#if defined LEAKY_0
                        // relu                                                                                                                                   
                        conv_ch_in.data[yy].lane[ll] = conv_final[ll][yy];
#elif defined LEAKY_0125 
                        // leaky relu
                        if(frac_relu<0){
                            conv_ch_in.data[yy].lane[ll] = conv_final[ll][yy]<<(-frac_relu);
                            if((conv_ch_in.data[yy].lane[ll] & 0x80) == 0x80){
                                conv_ch_in.data[yy].lane[ll] = 0x7f;
                            }
                            // #ifdef DEBUG_CONV
                            // if(yy==0 && k==(52*16*0+52*0+4))
                            //     printf("dot_cnt=%d yy=%d ll=%d   relu=%f (%x, %d)   sum_bias=%x (%d) \n", k, yy, ll, (float)(conv_ch_in.data[yy].lane[ll])*0.125, conv_ch_in.data[yy].lane[ll], conv_ch_in.data[yy].lane[ll], conv_final[ll][yy], conv_final[ll][yy]);
                            // #endif
                        }else{
                            conv_ch_in.data[yy].lane[ll] = (conv_final[ll][yy] + relu_add_coef[frac_relu+1])>>frac_relu;
                            // #ifdef DEBUG_CONV
                            // if(yy==0 && k==(52*16*0+52*0+4))
                            //     printf("dot_cnt=%d yy=%d ll=%d   relu=%f (%x, %d)   sum_bias=%x (%d) \n", k, yy, ll, (float)(conv_ch_in.data[yy].lane[ll])*0.125, conv_ch_in.data[yy].lane[ll], conv_ch_in.data[yy].lane[ll], conv_final[ll][yy], conv_final[ll][yy]);
                            // #endif
                        }
#endif
                    }
                } else{
                    conv_ch_in.data[yy].lane[ll] = conv_final[ll][yy];
                }

                #ifdef DEBUG_CONV
                if(ll==0 && k==0)
                    printf("dot_cnt=%d sum=%f rnd=%f sum_bias=%f final=%f (bias=%f)\n\n", k, (float)conv_out[ll], (float)conv_with_rnd_bit[ll], (float)conv_sum_bias[ll], (float)conv_final[ll], (float)bias[ll]);
                #endif
            }
        }

        write_channel_intel(conv_ch, conv_ch_in);
#endif
    }// end of output loop
    //printf("Kernel coreConv lanched !!!\n");
    
    // printf("\n%d\n", counter);
}



#if defined MEMWR_PORT
// Store Data to Global Memory
__kernel
__attribute__((task))
// __attribute__((reqd_work_group_size(1,1,LANE_NUM)))
void memWrite(
    // Params Ports
    uint output_num,
    ushort  out_dim1,
    ushort  out_dim2,
    ushort  gp_y,
    ushort  gp_z,
    ushort out_dim3,
    ushort out_dim1xbatch, // out_dim1 x sqrt(batch_size)
    uint   out_dim1x2xbatch, // out_dim1 x out_dim2 x batch_size
    uchar  batch_indx_dim1,
    uchar  batch_indx_dim2,
#if defined BN_FLOAT || defined RELU_FP
    uchar  bypass,        //0 read data form bn, 1 read data form conv
#endif
#ifdef USE_REORG
    uchar  layer_flag,
    uint    OFFSET,   // for yolov2 concat
#endif
#ifdef USE_REORG_CPU
    uint    OFFSET,   // for yolov2 concat
#endif
    uchar  pad_channel_offset,
    // uchar  pool_padding,	  // sometimes, pool need to padding
    // uchar  pool_on,
    // uchar  pool_size,
    // uchar  pool_stride,
    // Data Ports
    __global DPTYPE *restrict top
#ifdef USE_REORG
    ,__global DPTYPE *restrict route_buf
#endif
) {

    ushort global_x = 0;
    ushort global_y = 0;
    ushort global_z = 0; // max value 4096
    // uchar  local_z 	= 0; // max value 256
    ushort group_z  = 0;
    ushort group_y = 0;
    uchar local_z = 0;
    uchar local_y = 0;

    uchar  index_z_item; // max value 256
    ushort index_z_group;// max value 4096
    uint   top_addr;
#ifdef USE_REORG
    uint   top_addr_route;
#endif
    // bool pool_on_signal=1;
    DPTYPE_PE_SCAL   output;
    // __local DPTYPE buffer[PE_NUM_Y][LANE_NUM];


    for(uint i=0; i<output_num; i++){

        DPTYPE buffer[PE_NUM_Y][LANE_NUM];


        // if((pool_padding==1) && ((group_y==gp_y-1) || (global_x==out_dim1-1))) {
        //     #pragma unroll
        //     for(uchar ll=0; ll<LANE_NUM; ll++){ 
        //         output.data[PE_NUM_Y-1].lane[ll]=CZERO;
        //     }
        // } else {
#if defined BN_FLOAT || defined RELU_FP
            if(bypass==0) { //bypass==0,bn
                output = read_channel_intel(batchNorm_ch);
            } else // bypass == 1 bypass_bn_ch
                output = read_channel_intel(bypass_bn_ch);
#else
            output = read_channel_intel(conv_ch);
#endif
        // }

        // #pragma loop_coalesce
        #pragma unroll 1
        for(uchar gp_yz=0; gp_yz<PE_NUM_Y*LANE_NUM; gp_yz++){

                if(gp_yz==0){
                    #pragma unroll
                    for(uchar yy=0; yy<PE_NUM_Y; yy++){
                        #pragma unroll
                        for(uchar ll=0; ll<LANE_NUM; ll++) {
                            buffer[yy][ll]=output.data[yy].lane[ll];
                        }
                    }
                }

                // if((global_x==out_dim1-1) && (group_y==gp_y-1) && (group_z==gp_z-1))
                // printf("local: y=%d\tz=%d\n", local_y, local_z);

                global_z = group_z*LANE_NUM+local_z;
                global_y = group_y*PE_NUM_Y+local_y;
                index_z_group = (global_z-pad_channel_offset)/VEC_SIZE;
                index_z_item  = (global_z-pad_channel_offset)%VEC_SIZE;
                top_addr = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;

#ifdef USE_REORG
                if(layer_flag==12){
                    // index_z_group = (global_z-pad_channel_offset)/VEC_SIZE;
                    // index_z_item  = (global_z-pad_channel_offset)%VEC_SIZE;
                    top_addr_route = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                }
#endif

                // output dim3 in current layer may be larger than next layer (the value is changed to a value of multiples of LANE_NUM to saturated the wide pipeline input)
                // therefore, only write back the valid values without padding zeros
                if((global_z-pad_channel_offset)<out_dim3 && (global_z>=pad_channel_offset)) {
                    // 1. addressing expression with out batch processing is
                    // top[index_z_group*dim1*dim2*VEC_SIZE + global_y*dim1*VEC_SIZE + global_x*VEC_SIZE + index_z_item]=buffer[local_z];
                    // 2. addressing expression with batch processing (batch_size_in_dim = sqrt(batch_size)) is
                    // top[(index_z_group*out_dim2*out_dim1*batch_size_in_dim*batch_size_in_dim*VEC_SIZE + (global_y+batch_indx_dim2*out_dim2)*batch_size_in_dim*out_dim1*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item] = buffer[local_z];
                    // 3. simplified addressing with reduced cost of multipliers
                    //printf("b=%d\n",index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item);

#if (defined USE_REORG) || (defined USE_REORG_CPU)
                    // top[OFFSET+top_addr] = buffer[local_z][local_z];
                    // if(global_y<out_dim2)
                    top[OFFSET+top_addr] = buffer[local_y][local_z];
#else
                    // if(global_y<out_dim2)
                    top[top_addr] = buffer[local_y][local_z];
                    // top[top_addr] = buffer[local_z][local_y];
                    
#endif

#ifdef USE_REORG
                    // if(layer_flag==12 && global_y<out_dim2){
                    if(layer_flag==12){
                        route_buf[top_addr_route] = buffer[local_y][local_z];
                        // route_buf[top_addr_route] = buffer[local_z][local_y];
                    }
#endif

#ifdef DEBUG_MEMWR
                    if((global_z-pad_channel_offset) == 0 && global_x==0 && global_y>=13 && global_y<26){
                    //for(unsigned char ll=0; ll<LANE_NUM; ll++){
                      printf("MemWr results= %f (x=%d, y=%d, z=%d, ll=%d)\n", (float)output.lane[0].data[0], global_x, global_y, global_z, 0);
                    //}
                    }
#endif
                }

                if((local_y==PE_NUM_Y-1) && (local_z==LANE_NUM-1)){
                    local_y = 0;
                }else if(local_z==LANE_NUM-1){
                    local_y++;
                }
                if(local_z==LANE_NUM-1){
					local_z = 0;
                }else{
					local_z++;
                }
        }
        

        if((group_z==gp_z-1) && (group_y==gp_y-1) && (global_x==out_dim1-1)){// && (local_z==LANE_NUM-1)){
            group_z = 0;
        }else if((group_y==gp_y-1) && (global_x==out_dim1-1)){// && (local_z==LANE_NUM-1)){
            group_z++;
        }
        if((group_y==gp_y-1) && (global_x==out_dim1-1)){// && (local_z==LANE_NUM-1)){
            group_y = 0;
        }else if(global_x==out_dim1-1){// && (local_z==LANE_NUM-1)){
            group_y++;
        }
        if(global_x==out_dim1-1){// && (local_z==LANE_NUM-1)){
            global_x = 0;
        }else{//(local_z==LANE_NUM-1){
            global_x++;
        }
        // if(local_z==LANE_NUM-1){
        //     local_z = 0;
        // }else{
        //     local_z++;
        // }
        // if(local_y==PE_NUM_Y-1){
        //     local_y = 0;
        // }else{
        //     local_y++;
        // }
    }
}
#endif


#ifdef PE_NUM_Y_DIV_WR_2
// Store Data to Global Memory
__kernel
__attribute__((task))
// __attribute__((reqd_work_group_size(1,1,LANE_NUM)))
void memWrite(
    // Params Ports
    uint output_num, 
    ushort  out_dim1,
    ushort  out_dim2,
    ushort  gp_y,
    ushort  gp_z,
    ushort out_dim3,
    ushort out_dim1xbatch, // out_dim1 x sqrt(batch_size)
    uint   out_dim1x2xbatch, // out_dim1 x out_dim2 x batch_size
    uchar  batch_indx_dim1,
    uchar  batch_indx_dim2,
#if defined BN_FLOAT || defined RELU_FP
    uchar  bypass,        //0 read data form bn, 1 read data form conv
#endif
#ifdef USE_REORG
    uchar  layer_flag,
    uint    OFFSET,   // for yolov2 concat
#endif
#ifdef USE_REORG_CPU
    uint    OFFSET,   // for yolov2 concat
#endif
    uchar  pad_channel_offset,
    // uchar  pool_padding,	  // sometimes, pool need to padding
    // uchar  pool_on,
    // uchar  pool_size,
    // uchar  pool_stride,
    // Data Ports
    __global DPTYPE *restrict top0, // port 1
    __global DPTYPE *restrict top1, // port 2
    __global DPTYPE *restrict top2, // port 3
    __global DPTYPE *restrict top3, // port 4
    __global DPTYPE *restrict top4 // port 5
#ifdef MEMWR_MULTIPOR_6
    ,__global DPTYPE *restrict top5 // port 6
#endif
#ifdef MEMWR_MULTIPOR_7
    ,__global DPTYPE *restrict top6 // port 7
#endif
#ifdef MEMWR_MULTIPOR_8
    ,__global DPTYPE *restrict top7 // port 8
#endif
#ifdef MEMWR_MULTIPOR_9
    ,__global DPTYPE *restrict top8 // port 9
#endif
#ifdef USE_REORG
    ,__global DPTYPE *restrict route_buf0,
    __global DPTYPE *restrict route_buf1,
    __global DPTYPE *restrict route_buf2,
    __global DPTYPE *restrict route_buf3,
    __global DPTYPE *restrict route_buf4
#ifdef MEMWR_MULTIPOR_6
    ,__global DPTYPE *restrict route_buf5
#endif  
#ifdef MEMWR_MULTIPOR_7
    ,__global DPTYPE *restrict route_buf6
#endif  
#ifdef MEMWR_MULTIPOR_8
    ,__global DPTYPE *restrict route_buf7
#endif  
#ifdef MEMWR_MULTIPOR_9
    ,__global DPTYPE *restrict route_buf8  
#endif  
#endif
) {

    // bool pool_on_signal=1;
    DPTYPE_PE_SCAL   output;
    DPTYPE_SCAL   buffer0;
    DPTYPE_SCAL   buffer1;
    DPTYPE_SCAL   buffer2;
    DPTYPE_SCAL   buffer3;
    DPTYPE_SCAL   buffer4;
    DPTYPE_SCAL   buffer5;
    DPTYPE_SCAL   buffer6;
    DPTYPE_SCAL   buffer7;
    DPTYPE_SCAL   buffer8;
#ifdef MEMWR_MULTIPOR_6
    DPTYPE_SCAL   buffer9;
    DPTYPE_SCAL   buffer10;
#endif
#ifdef MEMWR_MULTIPOR_7
    DPTYPE_SCAL   buffer11;
    DPTYPE_SCAL   buffer12;
#endif
#ifdef MEMWR_MULTIPOR_8
    DPTYPE_SCAL   buffer13;
    DPTYPE_SCAL   buffer14;
#endif
#ifdef MEMWR_MULTIPOR_9
    DPTYPE_SCAL   buffer15;
    DPTYPE_SCAL   buffer16;
#endif


    // __local DPTYPE buffer[LANE_NUM][PE_NUM_Y];
    DPTYPE conv_out_tmp0, conv_out_tmp1, conv_out_tmp2, conv_out_tmp3, conv_out_tmp4; 
#ifdef MEMWR_MULTIPOR_6
    DPTYPE conv_out_tmp5; 
#endif
#ifdef MEMWR_MULTIPOR_7
    DPTYPE conv_out_tmp6;
#endif
#ifdef MEMWR_MULTIPOR_8 
    DPTYPE conv_out_tmp7;
#endif
#ifdef MEMWR_MULTIPOR_9 
    DPTYPE conv_out_tmp8;
#endif

    ushort global_x = 0;
    // ushort global_y = 0;
    ushort global_z = 0; // max value 4096
    // uchar  local_z 	= 0; // max value 256
    ushort group_z  = 0;
    ushort group_y = 0;
    uchar gp_itm_y, gp_itm_z;

    uchar  index_z_item; // max value 256
    ushort index_z_group;// max value 4096

    ushort global_y_port0 = 0;
    ushort global_y_port1 = 0;
    ushort global_y_port2 = 0;
    ushort global_y_port3 = 0;
    ushort global_y_port4 = 0;

    uint   top_addr_port0;
    uint   top_addr_port1;
    uint   top_addr_port2;
    uint   top_addr_port3;
    uint   top_addr_port4;

#ifdef MEMWR_MULTIPOR_6
    ushort global_y_port5 = 0;
    uint   top_addr_port5;
#endif
#ifdef MEMWR_MULTIPOR_7
    ushort global_y_port6 = 0;
    uint   top_addr_port6;
#endif
#ifdef MEMWR_MULTIPOR_8
    ushort global_y_port7 = 0;
    uint   top_addr_port7;
#endif
#ifdef MEMWR_MULTIPOR_9
    ushort global_y_port8 = 0;
    uint   top_addr_port8;
#endif


#ifdef USE_REORG
    uint   top_addr_route_port0;
    uint   top_addr_route_port1;
    uint   top_addr_route_port2;
    uint   top_addr_route_port3;
    uint   top_addr_route_port4;
#ifdef MEMWR_MULTIPOR_6
    uint   top_addr_route_port5;
#endif
#ifdef MEMWR_MULTIPOR_7
    uint   top_addr_route_port6;
#endif
#ifdef MEMWR_MULTIPOR_8
    uint   top_addr_route_port7;
#endif
#ifdef MEMWR_MULTIPOR_9
    uint   top_addr_route_port8;
#endif
#endif


    // printf("memWr: %d\n", output_num);
    
    for(uint i=0; i<output_num; i++){

        // if((pool_padding==1) && ((group_y==gp_y-1) || (global_x==out_dim1-1))) {
        //     #pragma unroll
        //     for(uchar ll=0; ll<LANE_NUM; ll++){ 
        //         output.data[PE_NUM_Y-1].lane[ll]=CZERO;
        //     }
        // } else {
#if defined BN_FLOAT || defined RELU_FP
            if(bypass==0) { //bypass==0,bn
                output = read_channel_intel(batchNorm_ch);
            } else // bypass == 1 bypass_bn_ch
                output = read_channel_intel(bypass_bn_ch);
#else
            output = read_channel_intel(conv_ch);
#endif
        // }


        // store the vectorized output into local buffer
        buffer0 = output.data[0];
        buffer1 = output.data[1];
        buffer2 = output.data[2];
        buffer3 = output.data[3];
        buffer4 = output.data[4];
        buffer5 = output.data[5];
        buffer6 = output.data[6];
        buffer7 = output.data[7];
        buffer8 = output.data[8];
#ifdef MEMWR_MULTIPOR_6
        buffer9 = output.data[9];
        buffer10 = output.data[10];
#endif
#ifdef MEMWR_MULTIPOR_7
        buffer11 = output.data[11];
        buffer12 = output.data[12];
#endif
#ifdef MEMWR_MULTIPOR_8
        buffer13 = output.data[13];
        buffer14 = output.data[14];
#endif
#ifdef MEMWR_MULTIPOR_9
        buffer15 = output.data[15];
        buffer16 = output.data[16];
#endif

        gp_itm_y = 0;
        gp_itm_z = 0;

        #pragma unroll 1
        for(uchar gp_itm_yz=0; gp_itm_yz<LANE_NUM*PE_NUM_Y_DIV_WR; gp_itm_yz++){
            // #pragma unroll 1
            // for(uchar local_y=0; local_y<PE_NUM_Y; local_y++){

                // if((global_x==out_dim1-1) && (group_y==gp_y-1) && (group_z==gp_z-1))
                // printf("local: y=%d\tz=%d\n", local_y, local_z);

                global_z = group_z*LANE_NUM+gp_itm_z;
                index_z_group = (global_z-pad_channel_offset)/VEC_SIZE;
                index_z_item  = (global_z-pad_channel_offset)%VEC_SIZE;

                global_y_port0 = group_y*PE_NUM_Y + gp_itm_y;
                global_y_port1 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR;
                global_y_port2 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR*2;
                global_y_port3 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR*3;
                global_y_port4 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR*4;
#ifdef MEMWR_MULTIPOR_6
                global_y_port5 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR*5;
#endif
#ifdef MEMWR_MULTIPOR_7
                global_y_port6 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR*6;
#endif
#ifdef MEMWR_MULTIPOR_8
                global_y_port7 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR*7;
#endif
#ifdef MEMWR_MULTIPOR_9
                global_y_port8 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR*8;
#endif

                top_addr_port0 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port0+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                top_addr_port1 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port1+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                top_addr_port2 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port2+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                top_addr_port3 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port3+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                top_addr_port4 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port4+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
#ifdef MEMWR_MULTIPOR_6       
                top_addr_port5 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port5+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
#endif
#ifdef MEMWR_MULTIPOR_7 
                top_addr_port6 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port6+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
#endif
#ifdef MEMWR_MULTIPOR_8
                top_addr_port7 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port7+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
#endif
#ifdef MEMWR_MULTIPOR_9
                top_addr_port8 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port8+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
#endif

#ifdef USE_REORG
                if(layer_flag==12){
                    // index_z_group = (global_z-pad_channel_offset)/VEC_SIZE;
                    // index_z_item  = (global_z-pad_channel_offset)%VEC_SIZE;
                    top_addr_route_port0 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port0+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                    top_addr_route_port1 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port1+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                    top_addr_route_port2 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port2+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                    top_addr_route_port3 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port3+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                    top_addr_route_port4 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port4+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
#ifdef MEMWR_MULTIPOR_6          
                    top_addr_route_port5 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port5+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
#endif
#ifdef MEMWR_MULTIPOR_7
                    top_addr_route_port6 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port6+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
#endif
#ifdef MEMWR_MULTIPOR_8
                    top_addr_route_port7 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port7+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
#endif
#ifdef MEMWR_MULTIPOR_9
                    top_addr_route_port8 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port8+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
#endif
                }
#endif

                // output dim3 in current layer may be larger than next layer (the value is changed to a value of multiples of LANE_NUM to saturated the wide pipeline input)
                // therefore, only write back the valid values without padding zeros
                if((global_z-pad_channel_offset)<out_dim3 && (global_z>=pad_channel_offset)) {
                    // 1. addressing expression with out batch processing is
                    // top[index_z_group*dim1*dim2*VEC_SIZE + global_y*dim1*VEC_SIZE + global_x*VEC_SIZE + index_z_item]=buffer[local_z];
                    // 2. addressing expression with batch processing (batch_size_in_dim = sqrt(batch_size)) is
                    // top[(index_z_group*out_dim2*out_dim1*batch_size_in_dim*batch_size_in_dim*VEC_SIZE + (global_y+batch_indx_dim2*out_dim2)*batch_size_in_dim*out_dim1*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item] = buffer[local_z];
                    // 3. simplified addressing with reduced cost of multipliers
                    //printf("b=%d\n",index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item);

                    switch(gp_itm_y)
                    {
                        case 0:
                        conv_out_tmp0 = buffer0.lane[gp_itm_z]; break;
                        case 1:
                        conv_out_tmp0 = buffer1.lane[gp_itm_z]; break;
                    }
                    // if(global_y_port0<out_dim2)
                    top0[OFFSET+top_addr_port0] = conv_out_tmp0;
                    // --- Port 2 ---
                    switch(gp_itm_y)
                    {
                        case 0:
                        conv_out_tmp1 = buffer2.lane[gp_itm_z]; break;
                        case 1:
                        conv_out_tmp1 = buffer3.lane[gp_itm_z]; break;
                    }
                    // if(global_y_port1<out_dim2)
                    top1[OFFSET+top_addr_port1] = conv_out_tmp1;
                    // --- Port 3 ---
                    switch(gp_itm_y)
                    {
                        case 0:
                        conv_out_tmp2 = buffer4.lane[gp_itm_z]; break;
                        case 1:
                        conv_out_tmp2 = buffer5.lane[gp_itm_z]; break;
                    }
                    // if(global_y_port2<out_dim2)
                    top2[OFFSET+top_addr_port2] = conv_out_tmp2;
                    // --- Port 4 ---
                    switch(gp_itm_y)
                    {
                        case 0:
                        conv_out_tmp3 = buffer6.lane[gp_itm_z]; break;
                        case 1:
                        conv_out_tmp3 = buffer7.lane[gp_itm_z]; break;
                    }
                    // if(global_y_port3<out_dim2)
                    top3[OFFSET+top_addr_port3] = conv_out_tmp3;
                    
#if (defined MEMWR_MULTIPOR_5) && (!defined MEMWR_MULTIPOR_6)
                    // --- Port 5 ---
                    // if(gp_itm_y==0 && global_y_port4<out_dim2){
                    if(gp_itm_y==0){
                        conv_out_tmp4 = buffer8.lane[gp_itm_z];
                        top4[OFFSET+top_addr_port4] = conv_out_tmp4;
                    }
#elif defined MEMWR_MULTIPOR_6
                    // --- Port 5 ---
                    switch(gp_itm_y)
                    {
                        case 0:
                        conv_out_tmp4 = buffer8.lane[gp_itm_z]; break;
                        case 1:
                        conv_out_tmp4 = buffer9.lane[gp_itm_z]; break;
                    }
                    // if(global_y_port4<out_dim2)
                    top4[OFFSET+top_addr_port4] = conv_out_tmp4;
#endif
#if (defined MEMWR_MULTIPOR_6) && (!defined MEMWR_MULTIPOR_7)
                    // --- Port 6 ---
                    // if(gp_itm_y==0 && global_y_port5<out_dim2){
                    if(gp_itm_y==0){
                        conv_out_tmp5 = buffer10.lane[gp_itm_z];
                        top5[OFFSET+top_addr_port5] = conv_out_tmp5;
                    }
#elif defined MEMWR_MULTIPOR_7
                    // --- Port 6 ---
                    switch(gp_itm_y)
                    {
                        case 0:
                        conv_out_tmp5 = buffer10.lane[gp_itm_z]; break;
                        case 1:
                        conv_out_tmp5 = buffer11.lane[gp_itm_z]; break;
                    }
                    // if(global_y_port5<out_dim2)
                    top5[OFFSET+top_addr_port5] = conv_out_tmp5;
#endif
#if (defined MEMWR_MULTIPOR_7) && (!defined MEMWR_MULTIPOR_8)
                    // --- Port 7 ---
                    // if(gp_itm_y==0 && global_y_port6<out_dim2){
                    if(gp_itm_y==0){
                        conv_out_tmp6 = buffer12.lane[gp_itm_z];
                        top6[OFFSET+top_addr_port6] = conv_out_tmp6;
                    }
#elif defined MEMWR_MULTIPOR_8
                    // --- Port 7 ---
                    switch(gp_itm_y)
                    {
                        case 0:
                        conv_out_tmp6 = buffer12.lane[gp_itm_z]; break;
                        case 1:
                        conv_out_tmp6 = buffer13.lane[gp_itm_z]; break;
                    }
                    // if(global_y_port6<out_dim2)
                    top6[OFFSET+top_addr_port6] = conv_out_tmp6;
#endif
#if (defined MEMWR_MULTIPOR_8) && (!defined MEMWR_MULTIPOR_9)
                    // --- Port 8 ---
                    // if(gp_itm_y==0 && global_y_port7<out_dim2){
                    if(gp_itm_y==0)
                    {
                        conv_out_tmp7 = buffer14.lane[gp_itm_z]; 
                        top7[OFFSET+top_addr_port7] = conv_out_tmp7;
                    }
#elif defined MEMWR_MULTIPOR_9
                    // --- Port 8 ---
                    switch(gp_itm_y)
                    {
                        case 0:
                        conv_out_tmp7 = buffer14.lane[gp_itm_z]; break;
                        case 1:
                        conv_out_tmp7 = buffer15.lane[gp_itm_z]; break;
                    }
                    // if(global_y_port7<out_dim2)
                    top7[OFFSET+top_addr_port7] = conv_out_tmp7; 
                    // --- Port 9 ---
                    // only one case
                    // if(gp_itm_y==0 && global_y_port8<out_dim2){
                    if(gp_itm_y==0){
                        conv_out_tmp8 = buffer16.lane[gp_itm_z];
                        top8[OFFSET+top_addr_port8] = conv_out_tmp8;
                    }
#endif


#ifdef USE_REORG
                    if(layer_flag==12){
                        // --- Port 1 ---
                        switch(gp_itm_y)
                        {
                            case 0:
                            conv_out_tmp0 = buffer0.lane[gp_itm_z]; break;
                            case 1:
                            conv_out_tmp0 = buffer1.lane[gp_itm_z]; break;
                        }
                        // if(global_y_port0<out_dim2)
                        route_buf0[top_addr_route_port0] = conv_out_tmp0;
                        // --- Port 2 ---
                        switch(gp_itm_y)
                        {
                            case 0:
                            conv_out_tmp1 = buffer2.lane[gp_itm_z]; break;
                            case 1:
                            conv_out_tmp1 = buffer3.lane[gp_itm_z]; break;
                        }
                        // if(global_y_port1<out_dim2)
                        route_buf1[top_addr_route_port1] = conv_out_tmp1;
                        // --- Port 3 ---
                        switch(gp_itm_y)
                        {
                            case 0:
                            conv_out_tmp2 = buffer4.lane[gp_itm_z]; break;
                            case 1:
                            conv_out_tmp2 = buffer5.lane[gp_itm_z]; break;
                        }
                        // if(global_y_port2<out_dim2)
                        route_buf2[top_addr_route_port2] = conv_out_tmp2;
                        // --- Port 4 ---
                        switch(gp_itm_y)
                        {
                            case 0:
                            conv_out_tmp3 = buffer6.lane[gp_itm_z]; break;
                            case 1:
                            conv_out_tmp3 = buffer7.lane[gp_itm_z]; break;
                        }
                        // if(global_y_port3<out_dim2)
                        route_buf3[top_addr_route_port3] = conv_out_tmp3;
#ifdef (defined MEMWR_MULTIPOR_5) && (!defined MEMWR_MULTIPOR_6)
                        // --- Port 5 ---
                        // if(gp_itm_y==0 && global_y_port4<out_dim2){
                        if(gp_itm_y==0){
                            conv_out_tmp4 = buffer8.lane[gp_itm_z];
                            route_buf4[top_addr_route_port4] = conv_out_tmp4;
                        }
#elif defined MEMWR_MULTIPOR_6
                        // --- Port 5 ---
                        switch(gp_itm_y)
                        {
                            case 0:
                            conv_out_tmp4 = buffer8.lane[gp_itm_z]; break;
                            case 1:
                            conv_out_tmp4 = buffer9.lane[gp_itm_z]; break;
                        }
                        // if(global_y_port4<out_dim2)
                        route_buf4[top_addr_route_port4] = conv_out_tmp4;
#endif
#if (defined MEMWR_MULTIPOR_6) && (!defined MEMWR_MULTIPOR_7)
                        // --- Port 6 ---
                        // if(gp_itm_y==0 && global_y_port5<out_dim2){
                        if(gp_itm_y==0){
                            conv_out_tmp5 = buffer10.lane[gp_itm_z];
                            route_buf5[top_addr_route_port5] = conv_out_tmp5;
                        }
#elif defined MEMWR_MULTIPOR_7
                        // --- Port 6 ---
                        switch(gp_itm_y)
                        {
                            case 0:
                            conv_out_tmp5 = buffer10.lane[gp_itm_z]; break;
                            case 1:
                            conv_out_tmp5 = buffer11.lane[gp_itm_z]; break;
                        }
                        // if(global_y_port5<out_dim2)
                        route_buf5[top_addr_route_port5] = conv_out_tmp5;
#endif
#if (defined MEMWR_MULTIPOR_7) && (!defined MEMWR_MULTIPOR_8)
                        // --- Port 7 ---
                        // if(gp_itm_y==0 && global_y_port6<out_dim2){
                        if(gp_itm_y==0){
                            conv_out_tmp6 = buffer12.lane[gp_itm_z];
                            route_buf6[top_addr_route_port6] = conv_out_tmp6;
                        }
#elif defined MEMWR_MULTIPOR_8
                        // --- Port 7 ---
                        switch(gp_itm_y)
                        {
                            case 0:
                            conv_out_tmp6 = buffer12.lane[gp_itm_z]; break;
                            case 1:
                            conv_out_tmp6 = buffer13.lane[gp_itm_z]; break;
                        }
                        // if(global_y_port6<out_dim2)
                        route_buf6[top_addr_route_port6] = conv_out_tmp6;
#endif
#if (defined MEMWR_MULTIPOR_8) && (!defined MEMWR_MULTIPOR_9)
                        // --- Port 8 ---
                        // if(gp_itm_y==0 && global_y_port7<out_dim2){
                        if(gp_itm_y==0)
                        {
                            conv_out_tmp7 = buffer14.lane[gp_itm_z]; 
                            route_buf7[top_addr_route_port7] = conv_out_tmp7;
                        }
#elif defined MEMWR_MULTIPOR_9
                        // --- Port 8 ---
                        switch(gp_itm_y)
                        {
                            case 0:
                            conv_out_tmp7 = buffer14.lane[gp_itm_z]; break;
                            case 1:
                            conv_out_tmp7 = buffer15.lane[gp_itm_z]; break;
                        }
                        // if(global_y_port7<out_dim2)
                        route_buf7[top_addr_route_port7] = conv_out_tmp7; 
                        // --- Port 9 ---
                        // only one case
                        // if(gp_itm_y==0 && global_y_port8<out_dim2){
                        if(gp_itm_y==0){
                            conv_out_tmp8 = buffer16.lane[gp_itm_z];
                            route_buf8[top_addr_route_port8] = conv_out_tmp8;
                        }    
                    }
#endif

#endif //#ifdef reorg

#ifdef DEBUG_MEMWR
                    if((global_z-pad_channel_offset) == 0 && global_x==0 && global_y>=13 && global_y<26){
                    //for(unsigned char ll=0; ll<LANE_NUM; ll++){
                      printf("MemWr results= %f (x=%d, y=%d, z=%d, ll=%d)\n", (float)output.lane[0].data[0], global_x, global_y, global_z, 0);
                    //}
                    }
#endif
                }
            // } // end local_y
                // virtual counters for group_item_x and group_item_y		
                if((gp_itm_y==PE_NUM_Y_DIV_WR-1) && (gp_itm_z==LANE_NUM-1)){
                    gp_itm_y = 0;
                }else if(gp_itm_z==LANE_NUM-1){
                    gp_itm_y++;
                }
                if(gp_itm_z==LANE_NUM-1){
					gp_itm_z = 0;
                }else{
					gp_itm_z++;
                }
        } //end local_z

        // if(global_x==0)
        // printf("global_x=%d\tgroup_y=%d\tgroup_z=%d\n", global_x, group_y, group_z);
        

        if((group_z==gp_z-1) && (group_y==gp_y-1) && (global_x==out_dim1-1)){// && (local_z==LANE_NUM-1)){
            group_z = 0;
        }else if((group_y==gp_y-1) && (global_x==out_dim1-1)){// && (local_z==LANE_NUM-1)){
            group_z++;
        }
        if((group_y==gp_y-1) && (global_x==out_dim1-1)){// && (local_z==LANE_NUM-1)){
            group_y = 0;
        }else if(global_x==out_dim1-1){// && (local_z==LANE_NUM-1)){
            group_y++;
        }
        if(global_x==out_dim1-1){// && (local_z==LANE_NUM-1)){
            global_x = 0;
        }else{//(local_z==LANE_NUM-1){
            global_x++;
        }
        // if(local_z==LANE_NUM-1){
        //     local_z = 0;
        // }else{
        //     local_z++;
        // }
        // if(local_y==PE_NUM_Y-1){
        //     local_y = 0;
        // }else{
        //     local_y++;
        // }
    }
}
#endif


#ifdef PE_NUM_Y_DIV_WR_1
// Store Data to Global Memory
__kernel
__attribute__((task))
// __attribute__((reqd_work_group_size(1,1,LANE_NUM)))
void memWrite(
    // Params Ports
    uint output_num,
    ushort  out_dim1,
    ushort  out_dim2,
    ushort  gp_y,
    ushort  gp_z,
    ushort out_dim3,
    ushort out_dim1xbatch, // out_dim1 x sqrt(batch_size)
    uint   out_dim1x2xbatch, // out_dim1 x out_dim2 x batch_size
    uchar  batch_indx_dim1,
    uchar  batch_indx_dim2,
#if defined BN_FLOAT || defined RELU_FP
    uchar  bypass,        //0 read data form bn, 1 read data form conv
#endif
#ifdef USE_REORG
    uchar  layer_flag,
    uint    OFFSET,   // for yolov2 concat
#endif
#ifdef USE_REORG_CPU
    uint    OFFSET,   // for yolov2 concat
#endif
    uchar  pad_channel_offset,
    // uchar  pool_padding,	  // sometimes, pool need to padding
    // uchar  pool_on,
    // uchar  pool_size,
    // uchar  pool_stride,
    // Data Ports
    __global DPTYPE *restrict top0, // port 1
    __global DPTYPE *restrict top1, // port 2
    __global DPTYPE *restrict top2, // port 3
    __global DPTYPE *restrict top3, // port 4
    __global DPTYPE *restrict top4,  // port 5
#ifdef MEMWR_MULTIPOR_9_1
    __global DPTYPE *restrict top5, // port 6
    __global DPTYPE *restrict top6, // port 7
    __global DPTYPE *restrict top7, // port 8
    __global DPTYPE *restrict top8 // port 9
#endif
#ifdef MEMWR_MULTIPOR_11
    ,__global DPTYPE *restrict top9, // port 10
    __global DPTYPE *restrict top10 // port 11
#endif
#ifdef MEMWR_MULTIPOR_13
    ,__global DPTYPE *restrict top11, // port 12
    __global DPTYPE *restrict top12 // port 13
#endif
#ifdef USE_REORG
    ,__global DPTYPE *restrict route_buf0,
    __global DPTYPE *restrict route_buf1,
    __global DPTYPE *restrict route_buf2,
    __global DPTYPE *restrict route_buf3,
    __global DPTYPE *restrict route_buf4,
#ifdef MEMWR_MULTIPOR_9_1
    __global DPTYPE *restrict route_buf5,
    __global DPTYPE *restrict route_buf6, 
    __global DPTYPE *restrict route_buf7,
    __global DPTYPE *restrict route_buf8
#endif
#ifdef MEMWR_MULTIPOR_11
    ,__global DPTYPE *restrict route_buf9,
    __global DPTYPE *restrict route_buf10 
#endif
#ifdef MEMWR_MULTIPOR_13
    ,__global DPTYPE *restrict route_buf11,
    __global DPTYPE *restrict route_buf12 
#endif     
#endif
) {

    // bool pool_on_signal=1;
    DPTYPE_PE_SCAL   output;
    DPTYPE_SCAL   buffer0;
    DPTYPE_SCAL   buffer1;
    DPTYPE_SCAL   buffer2;
    DPTYPE_SCAL   buffer3;
    DPTYPE_SCAL   buffer4;
#ifdef MEMWR_MULTIPOR_9_1
    DPTYPE_SCAL   buffer5;
    DPTYPE_SCAL   buffer6;
    DPTYPE_SCAL   buffer7;
    DPTYPE_SCAL   buffer8;
#endif
#ifdef MEMWR_MULTIPOR_11
    DPTYPE_SCAL   buffer9;
    DPTYPE_SCAL   buffer10;
#endif
#ifdef MEMWR_MULTIPOR_13
    DPTYPE_SCAL   buffer11;
    DPTYPE_SCAL   buffer12;
#endif


    // __local DPTYPE buffer[LANE_NUM][PE_NUM_Y];
    DPTYPE conv_out_tmp0, conv_out_tmp1, conv_out_tmp2, conv_out_tmp3, conv_out_tmp4; 
#ifdef MEMWR_MULTIPOR_9_1
    DPTYPE conv_out_tmp5, conv_out_tmp6, conv_out_tmp7, conv_out_tmp8;
#endif
#ifdef MEMWR_MULTIPOR_11  
    DPTYPE conv_out_tmp9, conv_out_tmp10;
#endif
#ifdef MEMWR_MULTIPOR_13
    DPTYPE conv_out_tmp11, conv_out_tmp12;
#endif


    ushort global_x = 0;
    // ushort global_y = 0;
    ushort global_z = 0; // max value 4096
    // uchar  local_z 	= 0; // max value 256
    ushort group_z  = 0;
    ushort group_y = 0;
    uchar gp_itm_y, gp_itm_z;

    ushort global_y_port0 = 0;
    ushort global_y_port1 = 0;
    ushort global_y_port2 = 0;
    ushort global_y_port3 = 0;
    ushort global_y_port4 = 0;
#ifdef MEMWR_MULTIPOR_9_1
    ushort global_y_port5 = 0;
    ushort global_y_port6 = 0;
    ushort global_y_port7 = 0;
    ushort global_y_port8 = 0;
#endif
#ifdef MEMWR_MULTIPOR_11 
    ushort global_y_port9 = 0;
    ushort global_y_port10 = 0;
#endif
#ifdef MEMWR_MULTIPOR_13
    ushort global_y_port11 = 0;
    ushort global_y_port12 = 0;
#endif

    uchar  index_z_item; // max value 256
    ushort index_z_group;// max value 4096

    uint   top_addr_port0;
    uint   top_addr_port1;
    uint   top_addr_port2;
    uint   top_addr_port3;
    uint   top_addr_port4;
#ifdef MEMWR_MULTIPOR_9_1
    uint   top_addr_port5;
    uint   top_addr_port6;
    uint   top_addr_port7;
    uint   top_addr_port8;
#endif
#ifdef MEMWR_MULTIPOR_11  
    uint   top_addr_port9;
    uint   top_addr_port10;
#endif
#ifdef MEMWR_MULTIPOR_13
    uint   top_addr_port11;
    uint   top_addr_port12;
#endif

#ifdef USE_REORG
    uint   top_addr_route_port0;
    uint   top_addr_route_port1;
    uint   top_addr_route_port2;
    uint   top_addr_route_port3;
    uint   top_addr_route_port4;
#ifdef MEMWR_MULTIPOR_9_1
    uint   top_addr_route_port5;
    uint   top_addr_route_port6;
    uint   top_addr_route_port7;
    uint   top_addr_route_port8;
#endif
#ifdef MEMWR_MULTIPOR_11 
    uint   top_addr_route_port9;
    uint   top_addr_route_port10;
#endif
#ifdef MEMWR_MULTIPOR_13
    uint   top_addr_route_port11;
    uint   top_addr_route_port12;
#endif
#endif


    // printf("memWr: %d\n", output_num);
    
    for(uint i=0; i<output_num; i++){

        // if((pool_padding==1) && ((group_y==gp_y-1) || (global_x==out_dim1-1))) {
        //     #pragma unroll
        //     for(uchar ll=0; ll<LANE_NUM; ll++){ 
        //         output.data[PE_NUM_Y-1].lane[ll]=CZERO;
        //     }
        // } else {
#if defined BN_FLOAT || defined RELU_FP
            if(bypass==0) { //bypass==0,bn
                output = read_channel_intel(batchNorm_ch);
            } else // bypass == 1 bypass_bn_ch
                output = read_channel_intel(bypass_bn_ch);
#else
            output = read_channel_intel(conv_ch);
#endif
        // }


        // store the vectorized output into local buffer
        buffer0 = output.data[0];
        buffer1 = output.data[1];
        buffer2 = output.data[2];
        buffer3 = output.data[3];
        buffer4 = output.data[4];
#ifdef MEMWR_MULTIPOR_9_1
        buffer5 = output.data[5];
        buffer6 = output.data[6];
        buffer7 = output.data[7];
        buffer8 = output.data[8];
#endif
#ifdef MEMWR_MULTIPOR_11  
        buffer9 = output.data[9];
        buffer10 = output.data[10];
#endif
#ifdef MEMWR_MULTIPOR_13
        buffer11 = output.data[11];
        buffer12 = output.data[12];
#endif

        gp_itm_y = 0;
        gp_itm_z = 0;

        #pragma unroll 1
        for(uchar gp_itm_yz=0; gp_itm_yz<LANE_NUM*PE_NUM_Y_DIV_WR; gp_itm_yz++){
            // #pragma unroll 1
            // for(uchar local_y=0; local_y<PE_NUM_Y; local_y++){

                // if((global_x==out_dim1-1) && (group_y==gp_y-1) && (group_z==gp_z-1))
                // printf("local: y=%d\tz=%d\n", local_y, local_z);

                global_z = group_z*LANE_NUM+gp_itm_z;
                index_z_group = (global_z-pad_channel_offset)/VEC_SIZE;
                index_z_item  = (global_z-pad_channel_offset)%VEC_SIZE;

                global_y_port0 = group_y*PE_NUM_Y + gp_itm_y;
                global_y_port1 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR;
                global_y_port2 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR*2;
                global_y_port3 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR*3;
                global_y_port4 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR*4;
#ifdef MEMWR_MULTIPOR_9_1
                global_y_port5 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR*5;
                global_y_port6 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR*6;
                global_y_port7 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR*7;
                global_y_port8 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR*8; 
#endif     
#ifdef MEMWR_MULTIPOR_11  
                global_y_port9 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR*9;
                global_y_port10 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR*10;
#endif
#ifdef MEMWR_MULTIPOR_13
                global_y_port11 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR*11;
                global_y_port12 = group_y*PE_NUM_Y + gp_itm_y + PE_NUM_Y_DIV_WR*12;
#endif

                top_addr_port0 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port0+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                top_addr_port1 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port1+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                top_addr_port2 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port2+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                top_addr_port3 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port3+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                top_addr_port4 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port4+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
#ifdef MEMWR_MULTIPOR_9_1
                top_addr_port5 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port5+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                top_addr_port6 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port6+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                top_addr_port7 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port7+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                top_addr_port8 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port8+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
#endif
#ifdef MEMWR_MULTIPOR_11  
                top_addr_port9 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port9+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                top_addr_port10 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port10+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
#endif
#ifdef MEMWR_MULTIPOR_13
                top_addr_port11 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port11+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                top_addr_port12 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port12+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
#endif

#ifdef USE_REORG
                if(layer_flag==12){
                    // index_z_group = (global_z-pad_channel_offset)/VEC_SIZE;
                    // index_z_item  = (global_z-pad_channel_offset)%VEC_SIZE;
                    top_addr_route_port0 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port0+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                    top_addr_route_port1 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port1+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                    top_addr_route_port2 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port2+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                    top_addr_route_port3 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port3+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                    top_addr_route_port4 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port4+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
#ifdef MEMWR_MULTIPOR_9_1
                    top_addr_route_port5 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port5+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                    top_addr_route_port6 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port6+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                    top_addr_route_port7 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port7+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                    top_addr_route_port8 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port8+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
#endif
#ifdef MEMWR_MULTIPOR_11 
                    top_addr_route_port9 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port9+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                    top_addr_route_port10 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port10+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
#endif
#ifdef MEMWR_MULTIPOR_13
                    top_addr_route_port11 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port11+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
                    top_addr_route_port12 = index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y_port12+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item;
#endif
                }
#endif

                // output dim3 in current layer may be larger than next layer (the value is changed to a value of multiples of LANE_NUM to saturated the wide pipeline input)
                // therefore, only write back the valid values without padding zeros
                if((global_z-pad_channel_offset)<out_dim3 && (global_z>=pad_channel_offset)) {
                    // 1. addressing expression with out batch processing is
                    // top[index_z_group*dim1*dim2*VEC_SIZE + global_y*dim1*VEC_SIZE + global_x*VEC_SIZE + index_z_item]=buffer[local_z];
                    // 2. addressing expression with batch processing (batch_size_in_dim = sqrt(batch_size)) is
                    // top[(index_z_group*out_dim2*out_dim1*batch_size_in_dim*batch_size_in_dim*VEC_SIZE + (global_y+batch_indx_dim2*out_dim2)*batch_size_in_dim*out_dim1*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item] = buffer[local_z];
                    // 3. simplified addressing with reduced cost of multipliers
                    //printf("b=%d\n",index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item);

                    // --- Port 1 ---
                    conv_out_tmp0 = buffer0.lane[gp_itm_z];
                    // if(global_y_port0<out_dim2)
                    top0[OFFSET+top_addr_port0] = conv_out_tmp0;
                        
                    // --- Port 2 ---
                    conv_out_tmp1 = buffer1.lane[gp_itm_z];
                    // if(global_y_port1<out_dim2)
                    top1[OFFSET+top_addr_port1] = conv_out_tmp1;    
                    
                    // --- Port 3 ---
                    conv_out_tmp2 = buffer2.lane[gp_itm_z];
                    // if(global_y_port2<out_dim2)
                    top2[OFFSET+top_addr_port2] = conv_out_tmp2;
                    
                    // --- Port 4 ---
                    conv_out_tmp3 = buffer3.lane[gp_itm_z];
                    // if(global_y_port3<out_dim2)
                    top3[OFFSET+top_addr_port3] = conv_out_tmp3;

                    // --- Port 5 ---
                    conv_out_tmp4 = buffer4.lane[gp_itm_z];
                    // if(global_y_port4<out_dim2)
                    top4[OFFSET+top_addr_port4] = conv_out_tmp4;
#ifdef MEMWR_MULTIPOR_9_1
                    // --- Port 6 ---
                    conv_out_tmp5 = buffer5.lane[gp_itm_z];
                    // if(global_y_port5<out_dim2)
                    top5[OFFSET+top_addr_port5] = conv_out_tmp5;    
                    
                    // --- Port 7 ---
                    conv_out_tmp6 = buffer6.lane[gp_itm_z];
                    // if(global_y_port6<out_dim2)
                    top6[OFFSET+top_addr_port6] = conv_out_tmp6;
                    
                    // --- Port 8 ---
                    conv_out_tmp7 = buffer7.lane[gp_itm_z];
                    // if(global_y_port7<out_dim2)
                    top7[OFFSET+top_addr_port7] = conv_out_tmp7;

                    // --- Port 9 ---
                    conv_out_tmp8 = buffer8.lane[gp_itm_z];
                    // if(global_y_port8<out_dim2)
                    top8[OFFSET+top_addr_port8] = conv_out_tmp8;
#endif
#ifdef MEMWR_MULTIPOR_11
                    // --- Port 10 ---
                    conv_out_tmp9 = buffer9.lane[gp_itm_z];
                    // if(global_y_port8<out_dim2)
                    top9[OFFSET+top_addr_port9] = conv_out_tmp9;

                    // --- Port 11 ---
                    conv_out_tmp10 = buffer10.lane[gp_itm_z];
                    // if(global_y_port8<out_dim2)
                    top10[OFFSET+top_addr_port10] = conv_out_tmp10;
#endif
#ifdef MEMWR_MULTIPOR_13
                    // --- Port 12 ---
                    conv_out_tmp11 = buffer11.lane[gp_itm_z];
                    // if(global_y_port8<out_dim2)
                    top11[OFFSET+top_addr_port11] = conv_out_tmp11;

                    // --- Port 13 ---
                    conv_out_tmp12 = buffer12.lane[gp_itm_z];
                    // if(global_y_port8<out_dim2)
                    top12[OFFSET+top_addr_port12] = conv_out_tmp12;
#endif


#ifdef USE_REORG
                    if(layer_flag==12){
                     // --- Port 1 ---
                    conv_out_tmp0 = buffer0.lane[gp_itm_z];
                    // if(global_y_port0<out_dim2)
                    route_buf0[top_addr_route_port0] = conv_out_tmp0;
                        
                    // --- Port 2 ---
                    conv_out_tmp1 = buffer1.lane[gp_itm_z];
                    // if(global_y_port1<out_dim2)
                    route_buf1[top_addr_route_port1] = conv_out_tmp1;    
                    
                    // --- Port 3 ---
                    conv_out_tmp2 = buffer2.lane[gp_itm_z];
                    // if(global_y_port2<out_dim2)
                    route_buf2[top_addr_route_port2] = conv_out_tmp2;
                    
                    // --- Port 4 ---
                    conv_out_tmp3 = buffer3.lane[gp_itm_z];
                    // if(global_y_port3<out_dim2)
                    route_buf3[top_addr_route_port3] = conv_out_tmp3;

                    // --- Port 5 ---
                    conv_out_tmp4 = buffer4.lane[gp_itm_z];
                    // if(global_y_port4<out_dim2)
                    route_buf4[top_addr_route_port4] = conv_out_tmp4;
#ifdef MEMWR_MULTIPOR_9_1
                    // --- Port 6 ---
                    conv_out_tmp5 = buffer5.lane[gp_itm_z];
                    // if(global_y_port5<out_dim2)
                    route_buf5[top_addr_route_port5] = conv_out_tmp5;    
                    
                    // --- Port 7 ---
                    conv_out_tmp6 = buffer6.lane[gp_itm_z];
                    // if(global_y_port6<out_dim2)
                    route_buf6[top_addr_route_port6] = conv_out_tmp6;
                    
                    // --- Port 8 ---
                    conv_out_tmp7 = buffer7.lane[gp_itm_z];
                    // if(global_y_port7<out_dim2)
                    route_buf7[top_addr_route_port7] = conv_out_tmp7;

                    // --- Port 9 ---
                    conv_out_tmp8 = buffer8.lane[gp_itm_z];
                    // if(global_y_port8<out_dim2)
                    route_buf8[top_addr_route_port8] = conv_out_tmp8;
#endif
#ifdef MEMWR_MULTIPOR_11
                    // --- Port 10 ---
                    conv_out_tmp9 = buffer9.lane[gp_itm_z];
                    // if(global_y_port7<out_dim2)
                    route_buf9[top_addr_route_port9] = conv_out_tmp9;

                    // --- Port 11 ---
                    conv_out_tmp10 = buffer10.lane[gp_itm_z];
                    // if(global_y_port8<out_dim2)
                    route_buf10[top_addr_route_port10] = conv_out_tmp10;
                    }
#endif
#ifdef MEMWR_MULTIPOR_13
                    // --- Port 12 ---
                    conv_out_tmp11 = buffer11.lane[gp_itm_z];
                    // if(global_y_port8<out_dim2)
                    route_buf11[top_addr_route_port11] = conv_out_tmp11;

                    // --- Port 13 ---
                    conv_out_tmp12 = buffer12.lane[gp_itm_z];
                    // if(global_y_port8<out_dim2)
                    route_buf12[top_addr_route_port12] = conv_out_tmp12;
#endif

#endif

#ifdef DEBUG_MEMWR
                    if((global_z-pad_channel_offset) == 0 && global_x==0 && global_y>=13 && global_y<26){
                    //for(unsigned char ll=0; ll<LANE_NUM; ll++){
                      printf("MemWr results= %f (x=%d, y=%d, z=%d, ll=%d)\n", (float)output.lane[0].data[0], global_x, global_y, global_z, 0);
                    //}
                    }
#endif
                }
            // } // end local_y
                // virtual counters for group_item_x and group_item_y		
                if((gp_itm_y==PE_NUM_Y_DIV_WR-1) && (gp_itm_z==LANE_NUM-1)){
                    gp_itm_y = 0;
                }else if(gp_itm_z==LANE_NUM-1){
                    gp_itm_y++;
                }
                if(gp_itm_z==LANE_NUM-1){
					gp_itm_z = 0;
                }else{
					gp_itm_z++;
                }
        } //end local_z

        // if(global_x==0)
        // printf("global_x=%d\tgroup_y=%d\tgroup_z=%d\n", global_x, group_y, group_z);
        

        if((group_z==gp_z-1) && (group_y==gp_y-1) && (global_x==out_dim1-1)){// && (local_z==LANE_NUM-1)){
            group_z = 0;
        }else if((group_y==gp_y-1) && (global_x==out_dim1-1)){// && (local_z==LANE_NUM-1)){
            group_z++;
        }
        if((group_y==gp_y-1) && (global_x==out_dim1-1)){// && (local_z==LANE_NUM-1)){
            group_y = 0;
        }else if(global_x==out_dim1-1){// && (local_z==LANE_NUM-1)){
            group_y++;
        }
        if(global_x==out_dim1-1){// && (local_z==LANE_NUM-1)){
            global_x = 0;
        }else{//(local_z==LANE_NUM-1){
            global_x++;
        }
        // if(local_z==LANE_NUM-1){
        //     local_z = 0;
        // }else{
        //     local_z++;
        // }
        // if(local_y==PE_NUM_Y-1){
        //     local_y = 0;
        // }else{
        //     local_y++;
        // }
    }
}
#endif



#ifdef USE_POOL
__kernel
__attribute__((task))
void maxPool(
		// Params Ports
		ushort data_dim1,       // input size
		ushort data_dim2,
		ushort data_dim3,
		uint   data_dim1xdim2,  // avoid one multiplier
		ushort pool_x,          // output size
		ushort pool_y,
        uint   pool_xy,
		uchar  pool_size,
		//uchar  pool_size_xy,    // pool_size_xy=pool_size_y*pool_size_x
		uchar  pool_stride,
		uchar  group_num_x,
		uchar  group_num_y,
		ushort group_num_xy,
        ushort  group_size_x,
		ushort  group_size_y,
		//ushort group_size_xy,
		ushort win_size_x,
		ushort win_size_y,
		uint  win_size_xy,
		__global  DPTYPE_VEC   *restrict bottom,
		__global  DPTYPE_VEC   *restrict top
){

	DPTYPE_VEC line_buf_0[POOL_WIN_SIZE];
	DPTYPE_VEC line_buf_1[POOL_WIN_SIZE];
	DPTYPE_VEC row_pool_reg;
	DPTYPE_VEC col_pool_reg;
	DPTYPE_VEC pool_reg[POOL_MAX_SIZE];
	DPTYPE_VEC pool_final;
	
    uchar  col_pool_cnt;
	uchar  row_pool_cnt;
	uchar group_cnt_x, group_cnt_y;
	uchar gp_item_cnt_x, gp_item_cnt_y;
	ushort win_itm_x, win_itm_y;
	
	ushort feature_idx_dim1, feature_idx_dim2, feature_idx_dim3;
	ushort out_dim1, out_dim2;
	uchar  xx, yy;
	
	bool  flag;  // ping-pong flag
	
	DPTYPE_VEC feature_vec;
	DPTYPE_VEC pool_reg_0, pool_reg_1, pool_reg_2, pool_reg_3;
	DPTYPE_VEC pool_pipe_reg[2];
	

	//flag = 0;
	CH_LOOP:for(unsigned short cc=0; cc<data_dim3; cc++){
		
		group_cnt_x = 0;
		group_cnt_y = 0;
		GP_LOOP:for(unsigned short group_cnt_xy=0; group_cnt_xy<group_num_xy; group_cnt_xy++){
			win_itm_x = 0;
			win_itm_y = 0;
			gp_item_cnt_x = 0;
			gp_item_cnt_y = 0;
			col_pool_cnt = 0;
			row_pool_cnt = 0;
			#pragma ivdep
			for(unsigned int win_itm_xy=0; win_itm_xy<win_size_xy; win_itm_xy++){ //14*14 for yolov2-tiny padding pool
				
				// Fetching data into line-buffer
				feature_idx_dim1 = win_itm_x+group_cnt_x*group_size_x*pool_stride; // win_itm_x + group_offset_x
				feature_idx_dim2 = win_itm_y+group_cnt_y*group_size_y*pool_stride; // win_itm_y + group_offset_y
				feature_idx_dim3 = cc;
				
				// for the last group, the group items whose input index are not within the input range are automatically discarded
				if((feature_idx_dim1<data_dim1)&&(feature_idx_dim2<data_dim2)){
					
					feature_vec = bottom[feature_idx_dim3*data_dim1xdim2 + feature_idx_dim2*data_dim1 + feature_idx_dim1];
#ifdef DEBUG_POOL
                //  if(cc==0)
                //     printf("win_x=%d y=%d\n", win_itm_x, win_itm_y);
				//if(group_cnt_x==0 && group_cnt_y==0){
				//	printf("Pooling: group_x_y=(%d, %d), x=%d, y=%d, z=%d, load feature=%f\n", group_cnt_x_pre, group_cnt_y_pre, win_itm_x, win_itm_y, cc, (float)feature_vec.data[0]);
				//	printf("Pooling: flag=%d, write addr=%d, data=%f\n", flag, (win_itm_y*win_size_x+win_itm_x), (float)feature_vec.data[0]);
				//}
#endif
				}
				else{
					#pragma unroll
					for(unsigned char vv=0; vv<VEC_SIZE; vv++){
						feature_vec.data[vv] = 0;
					}
                    // if(cc==0){
                    // printf("win_x=%d y=%d  padding\n", win_itm_x, win_itm_y);
                    // }

				}
				// Two line buffer to form the 3x3 pooling window
				// First read from line buffer for pooling and then write new line into the line buffer
				#pragma unroll
				for(unsigned char vv=0; vv<VEC_SIZE; vv++){
					// Max pooling among rows 
					// with the new value read from each line buffer
					if(pool_size==3)
						row_pool_reg.data[vv] = pool_max(line_buf_1[win_itm_x].data[vv], line_buf_0[win_itm_x].data[vv]);
					else // pool_size==2
						row_pool_reg.data[vv] = line_buf_0[win_itm_x].data[vv];
					
					pool_reg[0].data[vv] = pool_max(row_pool_reg.data[vv], feature_vec.data[vv]);
					
					// Max pooling among colums
					// with previous row-pooling results stored in shift-registers
					if(pool_size==3)
						col_pool_reg.data[vv] = pool_max(pool_reg[1].data[vv], pool_reg[2].data[vv]);
					else //pool_size==2
						col_pool_reg.data[vv] = pool_reg[1].data[vv];
					
					pool_final.data[vv] = pool_max(col_pool_reg.data[vv], pool_reg[0].data[vv]);
					
					// Update line buffer
					line_buf_1[win_itm_x].data[vv] = line_buf_0[win_itm_x].data[vv];
					line_buf_0[win_itm_x].data[vv] = feature_vec.data[vv];
					
					// Pushing the new row-pooling result into shift-registers
					#pragma unroll
					for(unsigned char p=POOL_MAX_SIZE-1; p>0; p--){
						pool_reg[p].data[vv]=pool_reg[p-1].data[vv];
					}
				}
				
#ifdef DEBUG_POOL
				printf("Maxpool: win_itm_xy=(%d,%d), line_buf_addr=%d, row_pool_cnt=%d, col_pool_cnt=%d\n", win_itm_x, win_itm_y, win_itm_x, row_pool_cnt, col_pool_cnt);
				//printf("         reg0=%f, reg1=%f, reg2=%f, max=%f\n", (float)pool_reg[0][0], (float)pool_reg[0][1], (float)pool_reg[0][2], (float)pool_final.lane[0]);
#endif
				
				// Correct max pooling is performed, result is write to external memory
				if(row_pool_cnt==(pool_size-1)&&col_pool_cnt==(pool_size-1)){
					
					out_dim1 = group_cnt_x*group_size_x + gp_item_cnt_x;
					out_dim2 = group_cnt_y*group_size_y + gp_item_cnt_y;
					// for the last group, the outputs whose input index are not within the output range are automatically discarded
					if((out_dim1<pool_x)&&(out_dim2<pool_y))
						top[cc*pool_xy + (group_cnt_y*group_size_y + gp_item_cnt_y)*pool_x + group_cnt_x*group_size_x + gp_item_cnt_x] = pool_final;
#ifdef DEBUG_POOL
					printf("Maxpool: gp_itm_xy=(%d,%d), pool_final=%f\n", gp_item_cnt_x, gp_item_cnt_y, (float)pool_final.data[0]);
#endif
				}

                // if(cc<=1)
                // printf("gp_x=%d\tgp_y=%d\twin_x=%d\twin_y=%d\trow=%d\tcol=%d\titem_x=%d\titem_y=%d\n", 
                // group_cnt_x, group_cnt_y, win_itm_x, win_itm_y, row_pool_cnt, col_pool_cnt, gp_item_cnt_x, gp_item_cnt_y);
				// printf("%d\t", win_itm_y);

				if(win_itm_x==win_size_x-1)
					gp_item_cnt_x = 0;
				else if(row_pool_cnt==(pool_size-1)&&col_pool_cnt==(pool_size-1))
					gp_item_cnt_x = gp_item_cnt_x + 1;
				
				if(row_pool_cnt==(pool_size-1)&&(win_itm_x==win_size_x-1))
					gp_item_cnt_y = gp_item_cnt_y + 1;
				
				// Generates pooling pipeline register wr/rd pointer
				if(row_pool_cnt==(pool_size-1)){
			    
					// For each time row_pool_cnt==(pool_size-1), waits for col_pool_cnt==(pool_size-1)
					// then, a correct pooling operation is ready to be performed
					// Pooling window slide counter for columns
					if(col_pool_cnt==(pool_size-1)){
						col_pool_cnt = (pool_size-pool_stride);
					}
					else
						col_pool_cnt = col_pool_cnt + 1;
				}
				// else
				// 	col_pool_cnt = 0;
				
				// Pooling window slide counter for rows
				if(row_pool_cnt==(pool_size-1)&&(win_itm_x==win_size_x-1)){
					row_pool_cnt = (pool_size-pool_stride);
                    col_pool_cnt = 0;
				}
				else if(win_itm_x==win_size_x-1)
					row_pool_cnt = row_pool_cnt + 1;
				
				// used as virtual loop counters
				if((win_itm_y==win_size_y-1) && (win_itm_x==win_size_x-1))
					win_itm_y = 0;
				else if(win_itm_x==win_size_x-1)
					win_itm_y++;
				
				if(win_itm_x==win_size_x-1)
					win_itm_x = 0;
				else
					win_itm_x++;
				
			} // end of win_item_xy

            // printf("%d\t%d\n", group_cnt_x, group_cnt_y);
			
			// virtual loop counters for current pooling window
			if((group_cnt_y==group_num_y-1) && (group_cnt_x==group_num_x-1))
				group_cnt_y = 0;
			else if(group_cnt_x==group_num_x-1)
				group_cnt_y++;
			
			if(group_cnt_x==group_num_x-1)
				group_cnt_x = 0;
			else
				group_cnt_x++;
			
		}// end of gp_xy
	}// end of channel

}
#endif



#if defined BN_FLOAT || defined RELU_FP
__kernel
__attribute__((task))
void batchNorm(
    uint input_num,//dim1*dim2*dim3/LANE_NUM
    uint dim1xdim2,
    ushort dim3,
    uchar  contol, //[0]-> relu  [1]->bypass pooling
    float frac2float,//conv out conver to float
    float frac2char//bn out conver to char
#ifndef RELU_FP 
    ,__global FLOAT_SCAL *restrict mean,
    __global FLOAT_SCAL *restrict var,
    __global FLOAT_SCAL *restrict alpha,
    __global FLOAT_SCAL *restrict beta
#endif
) {
    DPTYPE_PE_SCAL conv_ch_out;
    DPTYPE_PE_SCAL batchNorm_final;
    DPTYPE_PE_SCAL bn_ch_in;
    float bn_in;
    float bn_out;
    float sc_out;
    FLOAT_SCAL mean_ch;
    FLOAT_SCAL var_ch;
    FLOAT_SCAL alpha_ch;
    FLOAT_SCAL beta_ch;

    unsigned int iter=0;
    unsigned int j=dim1xdim2;

    DPTYPE out_final;
    float out_conver;

    // printf("BN: %d\n", input_num);

    for(unsigned int k=0; k<input_num; k++,j++) {
        conv_ch_out = read_channel_intel(conv_ch);
#ifndef RELU_FP 
        if(j==dim1xdim2) {
            mean_ch = mean[iter];
            var_ch = var[iter];
            alpha_ch = alpha[iter];
            beta_ch = beta[iter];
            iter=iter+1;
            j=0;
        }
#endif
        // #pragma unroll
        for(unsigned char yy=0; yy<PE_NUM_Y; yy++){
            for(unsigned char ll=0; ll<LANE_NUM; ll++) {
                // ll=part*LANE_NUM_DIV_PART_NUM+part_ll;
                // Convert DPTYPE fixed-point to float
                // Input data has "frac_dout" fractional bits
                bn_in = convert_float(conv_ch_out.data[yy].lane[ll])*frac2float;
#ifndef RELU_FP 
                bn_out=(bn_in-mean_ch.lane[ll])*var_ch.lane[ll];
                sc_out=bn_out*alpha_ch.lane[ll]+beta_ch.lane[ll];
                
                 // leaky Relu operation
                if(sc_out<0){
                    sc_out = sc_out*0.1;                          
                }
#else
              
                // leaky Relu operation
                if(bn_in<0){
                    sc_out = bn_in*0.125;
                    // sc_out = sc_out*0.125;
                }else{
                    sc_out = bn_in;
                }
#endif

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
                batchNorm_final.data[yy].lane[ll]=convert_char_rtz(out_conver);//Round towards zero

                // // Relu operation
                // if((contol&0x01)==0x01) {
                //     if((batchNorm_final.lane[ll]&MASKSIGN)==MASKSIGN) // MSB is sign bit
                //         bn_ch_in.lane[ll] = 0;
                //     else
                //         bn_ch_in.lane[ll] = batchNorm_final.lane[ll];
                // } else
                //     bn_ch_in.lane[ll] = batchNorm_final.lane[ll];

                bn_ch_in.data[yy].lane[ll] = batchNorm_final.data[yy].lane[ll];

#ifdef DEBUG_BN
                printf("ll=%d,conv_ch_out=%d,bn_in=%f,sc_out=%f,batchNorm_final.lane[ll]=%d,bn_ch_in.lane[ll]=%d\n",ll,conv_ch_out.lane[ll],bn_in,sc_out,batchNorm_final.lane[ll],bn_ch_in.lane[ll]);
#endif
            }
        }

        write_channel_intel(batchNorm_ch, bn_ch_in);
        //printf("Write channel item-%d is written in channel %d...\n", k, ll);
    }
    //printf("Kernel batchNorm lanched !!!\n");
}

#endif



#ifdef USE_ELTWISE
__kernel
__attribute__((task))
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
    __global DPTYPE_VEC *restrict bottom_1,
    __global DPTYPE_VEC *restrict bottom_2,
    __global DPTYPE_VEC *restrict top
) {
    DPTYPE_VEC data_out;
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
#pragma unroll
    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
        avgPoolSum[vv]=0;
#pragma unroll
        for(unsigned char pp=0; pp<ELT_PIPE_DEPTH; pp++) {
            avgPoolBuf[vv][pp]=0;
        }
    }

    for(unsigned int j=0; j<input_num; j++) {
#pragma unroll
        for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
            sum=bottom_1[j].data[vv]*in1_frac+bottom_2[j].data[vv]*in2_frac;
            // relu
            if(sum<0)
                sum=0;
            if(pool_on!=3) {
                sum=sum+0.5;
                // //overflow
                if(sum>127)
                    sum=127;
                data_out.data[vv]=convert_char_rtz(sum);//Round towards zero
            } else {
                sumAvg[vv]=sum+avgPoolBuf[vv][ELT_PIPE_DEPTH-1];
#pragma unroll
                for(uchar p=ELT_PIPE_DEPTH-1; p>0; p-- ) {
                    avgPoolBuf[vv][p] = avgPoolBuf[vv][p-1];
                }
                avgPoolBuf[vv][0]=sumAvg[vv];
            }

        }
        if(pool_on==0) {
            top[j]=data_out;
        } else if(pool_on==2) { //stride pool
            conv_itm_xyz=xyz_offset+xy_offset+conv_itm_x;
            if(conv_itm_xyz==j) {
                top[outnum]=data_out;
                outnum++;
                conv_itm_x=conv_itm_x+stride;
                if(conv_itm_x>=conv_x) {
                    conv_itm_x=0;
                    xy_offset+=stride*conv_x;
                    if(xy_offset>=conv_xy) {
                        xy_offset=0;
                        xyz_offset+=conv_xy;
                    }
                }
            }

        } else if(pool_on==3) { //avgpool
            ptr=ptr+1;
            if(ptr==AVGPOOL_SIZE) {
                ptr=0;
#pragma unroll
                for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
#pragma unroll
                    for(unsigned i=0; i<ELT_PIPE_DEPTH; i++) {
                        avgPoolSum[vv] += avgPoolBuf[vv][i];
                        avgPoolBuf[vv][i]=0;
                    }
                    out=avgPoolSum[vv]*divisor+0.5;
                    //overflow,because of relu no <0 value here
                    if(out>127)
                        out=127;
                    data_out.data[vv]=convert_char_rtz(out);//	Round towards zero
                    avgPoolSum[vv]=0;
                }
                top[outnum]=data_out;
                outnum++;
            }
        }

    }
    //printf("Kernel eltwise lanched !!!\n");
}
#endif


#ifdef USE_REORG
__kernel
__attribute__((task))
void reorg(
    //uint concat_in2_dim1x2x3_div_vec,
    uint reorg_dim1x2x3_div_vec, //13x13x256/vec = 26x26x64/vec
    uchar reorg_out_dim1, //13
    uchar reorg_out_dim2, //13
    uchar reorg_in_dim1, //26
    ushort reorg_in_dim1x2, //26x26
    uchar reorg_in_dim3_div_vec, //64/vec
    uchar reorg_dim3_out_div_in, //stride*stride
    uchar stride,
    __global DPTYPE_VEC *restrict input,
    __global DPTYPE_VEC *restrict output
){

    uchar w2, h2;
    uchar out_x = 0;
    uchar out_y = 0;
    ushort out_z = 0;
    uchar out_n = 0;

    for(uint j=0; j<reorg_dim1x2x3_div_vec; j++) {

        //c2 = out_z % in_dim3_div_lane; // 0, ..., out_dim3_div_lane
        //offset = out_z / in_dim3in_dim3_div_lane; //0,1,2,3
        //w2 = out_x * stride + offset % stride; //x 0,1
        //h2 = out_y * stride + offset / stride; //y 0,1
        //output[j] = input[c2*in_dim2*in_dim2+h2*reorg_in_dim1+w2];
        w2 = out_x * stride + out_n % stride; //x 0,1
        h2 = out_y * stride + out_n / stride; //y 0,1
        output[j] = input[out_z*reorg_in_dim1x2+h2*reorg_in_dim1+w2];
        //printf("%d\t", output[j].data[0]);

        //printf("n: %d\tz: %d\ty: %d\tx: %d\th2: %d\tw2 :%d\tidx: %d\todx: %d\n", out_n, out_z, out_y, out_x, h2, w2, out_z*reorg_in_dim1x2+h2*reorg_in_dim1+w2, j);

        if(out_n==reorg_dim3_out_div_in-1 && out_z==reorg_in_dim3_div_vec-1 && out_y==reorg_out_dim2-1 && out_x==reorg_out_dim1-1){
            out_n = 0;
        }else if(out_z==reorg_in_dim3_div_vec-1 && out_y==reorg_out_dim2-1 && out_x==reorg_out_dim1-1){
            out_n = out_n + 1;
            //printf("\n");
        }
        if(out_z==reorg_in_dim3_div_vec-1 && out_y==reorg_out_dim2-1 && out_x==reorg_out_dim1-1){
            out_z = 0;
        }else if(out_y==reorg_out_dim2-1 && out_x==reorg_out_dim1-1){
            out_z = out_z + 1;
        }
        if(out_y==reorg_out_dim2-1 && out_x==reorg_out_dim1-1){
            out_y = 0;
        }else if(out_x==reorg_out_dim1-1){
            out_y = out_y + 1;
            //printf("\n");
        }
        if(out_x==reorg_out_dim1-1){
            out_x = 0;
        }else{
            out_x = out_x + 1;
        }
    }
}
#endif


#ifdef USE_LRN
__kernel
__attribute__((max_work_group_size(LRN_MAX_LOCAL_SIZE)))
void lrn(
    // Params Ports
    uchar data_dim1,
    uchar data_dim2,
    char  frac_dout,
    // Data Ports
    __global DPTYPE_VEC *restrict bottom,
    __global DPTYPE_VEC *restrict top
) {
    uchar  global_x = get_global_id(0); // max value 256
    uchar  global_y = get_global_id(1); // max value 256
    ushort global_z = get_global_id(2); // max value 4096

#ifdef DEBUG_LRN
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);
    int local_z = get_local_id(2);
    int block_x = get_group_id(0);
    int block_y = get_group_id(1);
    int block_z = get_group_id(2);
#endif

    __local DPTYPE z_buffer[VEC_SIZE*LRN_MAX_LOCAL_SIZE+LRN_WIN_SIZE]; // allocate two more points for padding
    __local DPTYPE lrn_buffer[VEC_SIZE*LRN_MAX_LOCAL_SIZE];
    DPTYPE_SCAL data_in;
    DPTYPE_SCAL data_pad_left;
    DPTYPE_SCAL data_pad_right;
    DPTYPE_SCAL data_out;
    DPTYPE_VEC    data_in_partial;
    DPTYPE_VEC    data_left_partial;
    DPTYPE_VEC    data_right_partial;
    DPTYPE_VEC    data_out_partial;
    int          *convert_ptr;
    int          expo;
    uint         manti;
    uint         addr_1, addr_2, addr;
    float        lrn_reg1, lrn_reg2, lrn_tmp, lrn_out;
    short        lrn_cnvt, lrn_cnvt2;

    // Load the all data in one line along dim3 into local line buffer
#pragma unroll
    for(unsigned char ll=0; ll<VEC_SIZE; ll++) {
        z_buffer[global_z*VEC_SIZE+ll+LRN_WIN_SIZE/2] = bottom[global_z*data_dim2*data_dim1 + global_y*data_dim1+ global_x].data[ll];
    }

    //Padding left
    if(global_z==0) {
#pragma unroll
        for(unsigned char ll=0; ll<LRN_WIN_SIZE/2; ll++) {
            z_buffer[ll] = CZERO;
        }
    }

    // Padding right
    if(global_z==(get_global_size(2)-1)) {
#pragma unroll
        for(unsigned char ll=0; ll<LRN_WIN_SIZE/2; ll++) {
            z_buffer[VEC_SIZE*get_local_size(2)+ll+LRN_WIN_SIZE/2] = CZERO;
        }
    }

#ifdef DEBUG_LRN
    if(global_z==0&&global_x==0&&global_y==0)
        printf("Kernel LRN: work-item x=%d, y=%d, z=%d(z_local=%d)\n", global_x, global_y, global_z, local_z);
#endif
    barrier(CLK_LOCAL_MEM_FENCE); // fill all values of the line bufer before reading it

    // Piecewise interpolation pipeline for lrn operation (i.e., y=pwlf(x'))
    for(unsigned char ll=0; ll<VEC_SIZE; ll++) {
        // First Step: Coefficients table looking-up
        // Calculate x'=sum(x(k)^2) for the pwlf function, x(k)s are from adjacent featuremaps
        lrn_reg2 = CZERO;
#pragma unroll
        for(char k=-LRN_WIN_SIZE/2; k<=LRN_WIN_SIZE/2; k++) {
            // Convert DPTYPE fixed-point to float
            // Input data has "frac_dout" fractional bits
            // Note: current version only support frac_dout<0
            lrn_cnvt = z_buffer[global_z*VEC_SIZE+ll+k+LRN_WIN_SIZE/2]<<(-frac_dout);
            lrn_reg1 = convert_float(lrn_cnvt);
            lrn_reg2 += lrn_reg1 * lrn_reg1;
#ifdef DEBUG_LRN
            if(global_z==0&&global_x==0&&global_y==0)
                printf("x=%f(k=%d), ", lrn_reg1, k);
#endif
        }
        // Get the exponent and mantissa of x' (assuming x'>=0)
        convert_ptr = (int*) (&lrn_reg2);
        // substract the bias 127 to get exponent
        expo = (EXP_MASK & (*convert_ptr >> MAN_BITS)) - 127;
        manti = ((*convert_ptr) & MAN_MASK); //does not include the implicit 1

        // Get level-1 table item (segment) index from exponent
        addr_1 = ((expo-EXP_STEP_MIN)>>EXP_STEP_LOG)<<MAN_INDEX_BITS;
        // Get level-2 table item (segment) index from mantissa
        addr_2 = (manti>>(MAN_BITS-MAN_INDEX_BITS) & MAN_INDEX_MASK)+1;
        // Combine level-1 and level-2 table index to get true table address
        if(expo<EXP_STEP_MIN)
            addr = 0; // use the first segment
        else
            addr = addr_1+addr_2;

        // Sencond Step: Perform piecewise linear interpolation
        lrn_tmp = ((lrn_reg2-x_sample[addr])*h_inv[addr])*coef1[addr] + coef0[addr];

        // Third Step: Perform lrn operation
        lrn_cnvt2 = z_buffer[global_z*VEC_SIZE+ll+LRN_WIN_SIZE/2]<<(-frac_dout);
        lrn_out = lrn_tmp*convert_float(lrn_cnvt2);

        // Convert float to DPTYPE fixed-point
        // Note: current version only support frac_din=0 for next layer
        lrn_buffer[global_z*VEC_SIZE+ll] = convert_char_rte(lrn_out);

#ifdef DEBUG_LRN
        if(global_z==0&&global_x==0&&global_y==0)
            printf("\nKernel LRN (ll=%d): pwlf_x=%f, expo=%d, addr=%d, pwlf_y=%f, lrn=%f\n", ll, lrn_reg2, expo, addr, lrn_tmp, lrn_out);
#endif
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the results back to global mem
#pragma unroll
    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
        data_out_partial.data[vv]=lrn_buffer[global_z*VEC_SIZE+vv];
    }
    top[global_z*data_dim2*data_dim1 + global_y*data_dim1 + global_x] = data_out_partial;

#ifdef DEBUG_LRN_OUT
    if(global_z==0&&global_x==0&&global_y==0)
        printf("\nKernel LRN OUT: x=%d, y=%d, z=%d, result=%f\n", global_x, global_y, global_z, (float)data_out_partial.data[0]);
#endif

}
#endif
