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
//#define DEBUG_POOL
//#define DEBUG_MEMWR
//#define DEBUG_LRN
//#define DEBUG_LRN_OUT

#include "hw_param.cl"
#include "pipe.cl" // for xilinx, the channels are defined here by using pipes

// Define the precision of the data-path
typedef char DPTYPE;
typedef int  MACTYPE;

// Vectorized data type
typedef struct {
    DPTYPE data[VEC_SIZE];
} lane_data;

// Combined vec-data type from multiple lane
typedef struct {
    lane_data lane[LANE_NUM];
} channel_vec;

// Combined scalar data type from multiple lane
typedef struct {
    DPTYPE lane[LANE_NUM];
} channel_scal;


// parallel MAC units including (VEC_SIZE-1) multipliers
__attribute__((always_inline))
MACTYPE mac(lane_data input, lane_data weights)
{
    MACTYPE output = MASK_MULT & CZERO;
    __attribute__((opencl_unroll_hint))
    for(int i=0; i<VEC_SIZE; i++) {
        output += input.data[i]*weights.data[i];
    }
    return output;
}

__attribute__((always_inline))
DPTYPE pool_max(DPTYPE a_in, DPTYPE b_in)
{
    DPTYPE max_value;

    if(a_in >= b_in)
        max_value = a_in;
    else
        max_value = b_in;

    return max_value;

}

// Fetch Data from Global Memory
__kernel
__attribute__((reqd_work_group_size(1,1,1)))
//__attribute__((xcl_dataflow))
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
    //uint   item_loop_bound,
    // Data Ports
    __global lane_data    *restrict bottom   //__attribute__((xcl_data_pack(bottom))),
    ,__global channel_vec  *restrict weights //__attribute__((xcl_data_pack(weights))),
    ,__global channel_scal *restrict bias   )// __attribute__((xcl_data_pack(bias)))    )

{


    // Input Data, Weights and Bias
    lane_data     data_vec       ;
    channel_vec   data_ch_vec    ;
    channel_vec   weight_ch_vec  ;
    channel_scal  bias_ch_in     ;
    ushort        data_offset = 0; // assuming the 1st layer is not in split

    // virtual loop counters
    ushort gp_num_x, gp_num_y, out_idx_z;
    ushort gp_num_x_winbuf, gp_num_y_winbuf, out_idx_z_winbuf;
    uchar  output_idx_dim1, output_idx_dim2;
    ushort output_idx_dim3;
    uchar  win_itm_x, win_itm_y;
    ushort win_itm_z;

    uchar  gp_item_idx_x;

    ushort feature_idx_dim1, feature_idx_dim2;
    ushort feature_idx_dim3;
    uint   item_loop_bound;
    uchar  flag; // ping-pong flag

    // Ping-pong buffer
    __local lane_data win_buffer[2][WIN_BUF_SIZE]  __attribute__((xcl_array_partition(complete,1))); // working sequence 0->1->0->1 ...
    // Weight buffer
    __local channel_vec weight_buffer[WEIGHT_BUF_SIZE] ;
Init:
    for(unsigned short win_itm_z=0; win_itm_z<weight_dim3/VEC_SIZE; win_itm_z++) {
        for(unsigned char  win_itm_y=0; win_itm_y<win_size_y; win_itm_y++) {
            for(unsigned char  win_itm_x=0; win_itm_x<win_size_x; win_itm_x++) {

                feature_idx_dim1 = win_itm_x;
                feature_idx_dim2 = win_itm_y;
                feature_idx_dim3 = win_itm_z;

                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2>=padding && feature_idx_dim2<data_dim2+padding))

                    data_vec = bottom[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2-padding)*data_dim1 + (feature_idx_dim1-padding)];
                else {
                    __attribute__((opencl_unroll_hint))
                    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                        data_vec.data[vv] = CZERO;
                    }
                }
                win_buffer[0][win_itm_z*win_size_y*win_size_x + win_itm_y*win_size_x + win_itm_x] = data_vec;
            }
        }
    }

    if(group_num_x==1)
        gp_num_x_winbuf = 0; // there is only one group for FC mode when batch=1
    else
        gp_num_x_winbuf = 1; // loop start from the second group
    gp_num_y_winbuf = 0;
    out_idx_z_winbuf = 0;

    // reset global group virtual loop counters
    gp_num_x = 0;
    gp_num_y = 0;
    out_idx_z = 0;

Group:
    for(unsigned int out_idx_xyz=0; out_idx_xyz<(weight_dim4_div_lane*group_num_y*group_num_x); out_idx_xyz++) {

        if(split==0)
            data_offset = 0;
        else if(out_idx_z_winbuf<(weight_dim4_div_lane>>1)) // the lower half of the output feature maps depend on the lower half of the input
            data_offset = 0;
        else
            data_offset = weight_dim3/VEC_SIZE;	// the upper half of the output feature maps depend on the upper half of the input

        flag = out_idx_xyz & 0x01; //ping-pong flag

        // reset output loop counters
        output_idx_dim1 = 0;
        output_idx_dim2 = 0;
        output_idx_dim3 = 0;
        // reset in-group item counters
        gp_item_idx_x = 0;

        // reset input winbuffer loop counters
        win_itm_x = 0;
        win_itm_y = 0;
        win_itm_z = 0;

        if(gp_num_x==group_num_x-1)
            item_loop_bound = win_size_x>=group_rem_size_x ? (win_size_xyz/VEC_SIZE) : (group_rem_size_xyz/VEC_SIZE);
        else
            item_loop_bound = (weight_dim1x2x3*CONV_GP_SIZE_Y*CONV_GP_SIZE_X/VEC_SIZE);
Item:
        for(unsigned int  win_itm_xyz=0; win_itm_xyz<item_loop_bound; win_itm_xyz++) {
            // Winbuffer loading operations
            if(win_itm_z<weight_dim3/VEC_SIZE) {

                feature_idx_dim1 = win_itm_x+gp_num_x_winbuf*CONV_GP_SIZE_X*stride;
                feature_idx_dim2 = win_itm_y+gp_num_y_winbuf*CONV_GP_SIZE_Y*stride;
                feature_idx_dim3 = win_itm_z;

                if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2>=padding && feature_idx_dim2<data_dim2+padding))
                    data_vec = bottom[data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2-padding)*data_dim1 + (feature_idx_dim1-padding)];
                else {
                    __attribute__((opencl_unroll_hint))
                    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                        data_vec.data[vv] = CZERO;
                    }
                }

                win_buffer[(~flag)&0x01][win_itm_z*win_size_y*win_size_x + win_itm_y*win_size_x + win_itm_x] = data_vec;
                // used as loop counters
                if((win_itm_z==weight_dim3/VEC_SIZE-1) && (win_itm_y==win_size_y-1) && (win_itm_x==win_size_x-1))
                    win_itm_z = 0;
                else if((win_itm_y==win_size_y-1) && (win_itm_x==win_size_x-1))
                    win_itm_z++;

                if((win_itm_y==win_size_y-1) && (win_itm_x==win_size_x-1))
                    win_itm_y = 0;
                else if(win_itm_x==win_size_x-1)
                    win_itm_y++;

                if(win_itm_x==win_size_x-1)
                    win_itm_x = 0;
                else
                    win_itm_x++;

            }

            // Load weight into weight buffer
            if(gp_item_idx_x==0) {
                weight_ch_vec = weights[out_idx_z*weight_dim1x2x3/VEC_SIZE + output_idx_dim3*weight_dim1x2 + output_idx_dim2*weight_dim1 + output_idx_dim1];
                weight_buffer[output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1] = weight_ch_vec;
            }

            // In this version, grouping is only performed in row (x) direction
            if(gp_num_x*CONV_GP_SIZE_X+gp_item_idx_x<conv_x) {

                if(output_idx_dim1==0 && output_idx_dim2==0 && output_idx_dim3==0) {
                    bias_ch_in = bias[out_idx_z];
                    bias_ch_write_pipe_block(bias_ch_in);
                    //#ifdef DEBUG_MEMRD
                    //printf("work-item x=%d, y=%d, z=%d, channel =0, write bias=%f\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, bias_ch_in.lane[0]);
                    //#endif
                }
                // data
                data_vec = win_buffer[flag][output_idx_dim3*win_size_y*win_size_x + output_idx_dim2*win_size_x + (output_idx_dim1+gp_item_idx_x*stride)];
                __attribute__((opencl_unroll_hint))
                for(unsigned char ll=0; ll<LANE_NUM; ll++) {
                    data_ch_vec.lane[ll] = data_vec;
                }
                data_write_pipe_block(data_ch_vec);

                weight_ch_vec = weight_buffer[output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1];
                weight_write_pipe_block(weight_ch_vec);

#ifdef DEBUG_MEMRD
                printf("work-item x=%d, y=%d, z=%d, offset=%d, write data in channel 0=%f\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, data_offset, (float)data_ch_vec.lane[0].data[0]);
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

            if(win_itm_xyz == item_loop_bound-1) {

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

            }
        }

    }

    //printf("Kernel 0 lanched !!!\n");
}

__kernel
__attribute__((reqd_work_group_size(1,1,1)))
//__attribute__((xcl_dataflow))
void coreConv(
    // Params Ports
    uint  output_num,
    uint  conv_loop_cnt,
    uint  contol, //[0]-> relu  [1]->bypass pooling
    char  frac_w,
    char  frac_din,
    char  frac_dout
)
{
    channel_vec mac_data;
    channel_vec mac_weight;
    channel_scal bias_ch_out;
    channel_scal conv_ch_in;
    DPTYPE  bias[LANE_NUM];
    MACTYPE conv_out[LANE_NUM];
    MACTYPE lane_accum[LANE_NUM];
    MACTYPE accum_piped[LANE_NUM*PIPE_DEPTH] __attribute__((xcl_array_partition(complete,1))) ;
    MACTYPE conv_sign_exten[LANE_NUM];
    MACTYPE conv_with_rnd_bit[LANE_NUM];
    MACTYPE conv_sum_bias[LANE_NUM];
    DPTYPE  conv_final[LANE_NUM];

    // each iteration generates one output
loop1:
    for(unsigned int k=0; k<output_num; k++) {

loop2:
        for(int j=0; j<conv_loop_cnt; j++) {

            if(j == 0) {
                bias_ch_read_pipe_block(bias_ch_out);

                __attribute__((opencl_unroll_hint))
                for(unsigned char ll=0; ll<LANE_NUM; ll++) {

                    conv_out[ll] = CZERO;
                    bias[ll] = bias_ch_out.lane[ll];

                    __attribute__((opencl_unroll_hint))
                    for(unsigned int p=0; p<PIPE_DEPTH; p++) {
                        accum_piped[ll*PIPE_DEPTH+p] = MASK_ACCUM & CZERO;
                    }
                }

            }


            // load data and weights for each lane
            data_read_pipe_block(mac_data);
            weight_read_pipe_block(mac_weight);

            __attribute__((opencl_unroll_hint))
            for(unsigned char ll=0; ll<LANE_NUM; ll++) {

                lane_accum[ll] = (MASK_ACCUM & accum_piped[ll*PIPE_DEPTH+PIPE_DEPTH-1]) + (MASK_MULT & mac(mac_data.lane[ll], mac_weight.lane[ll]));

                __attribute__((opencl_unroll_hint))
                for(unsigned int p=PIPE_DEPTH-1; p>0; p-- ) {
                    accum_piped[ll*PIPE_DEPTH+p] = MASK_ACCUM & accum_piped[ll*PIPE_DEPTH+p-1];
                }

                accum_piped[ll*PIPE_DEPTH] = MASK_ACCUM & lane_accum[ll];

#ifdef DEBUG_CONV
                //if(ll==0 && k==0){
                //	printf("dot_cnt=%d data=%f weight=%f (loop=%d, lane= %d, vec=0)\n", k, (float)mac_data.lane[ll].data[0], (float)mac_weight.lane[ll].data[0], j, ll);
                //}
#endif
            }

            if(j == conv_loop_cnt-1) {
                __attribute__((opencl_unroll_hint))
                for(unsigned char ll=0; ll<LANE_NUM; ll++) {
                    __attribute__((opencl_unroll_hint))
                    for(unsigned i=0; i<PIPE_DEPTH; i++) {
                        conv_out[ll] += accum_piped[ll*PIPE_DEPTH+i];
                    }

                    if(conv_out[ll]>=0)
                        conv_sign_exten[ll] = 0x00;
                    else
                        conv_sign_exten[ll] = ~(0xFFFFFFFF>>(frac_w+frac_din-frac_dout-1));

                    conv_with_rnd_bit[ll] = (conv_sign_exten[ll] | (conv_out[ll]>>(frac_w+frac_din-frac_dout-1))) + 0x01;

                    if(conv_with_rnd_bit[ll]>=256)
                        conv_sum_bias[ll] = MASK9B & 0xFF;
                    else if(conv_with_rnd_bit[ll]<-256)
                        conv_sum_bias[ll] = MASK9B & 0x100;
                    else
                        conv_sum_bias[ll] = (MASK9B & conv_with_rnd_bit[ll])+(bias[ll]>>(frac_w-frac_dout-1))+0x01;

                    conv_final[ll] = MASK8B & (conv_sum_bias[ll]>>0x01);

                    // Relu operation
                    if((contol&0x01)==0x01) {
                        if((conv_final[ll]&MASKSIGN)==MASKSIGN)
                            conv_ch_in.lane[ll] = 0;
                        else
                            conv_ch_in.lane[ll] = conv_final[ll];
                    } else
                        conv_ch_in.lane[ll] = conv_final[ll];
                }

                // write convoluation results
                if((contol&0x02)==0x02) {
                    //by-pass pooling
                    bypass_ch_write_pipe_block(conv_ch_in);
                } else { // to pooling kernel
                    conv_ch_write_pipe_block(conv_ch_in);
                }
                //printf("Write channel item-%d is written in channel %d...\n", k, ll);

            }
        }// end of conv loop

#ifdef DEBUG_CONV
        if(ll==0 && k==0)
            printf("dot_cnt=%d sum=%f rnd=%f sum_bias=%f final=%f (bias=%f)\n\n", k, (float)conv_out[ll], (float)conv_with_rnd_bit[ll], (float)conv_sum_bias[ll], (float)conv_final[ll], (float)bias[ll]);
#endif

    }// end of output loop

}


__kernel
__attribute__((reqd_work_group_size(1,1,1)))
//__attribute__((xcl_dataflow))
void maxPool(
    // Params Ports
    uint  input_num,
    uchar line_size,  // line_size should be no larger than POOL_LBUF_DEPTH
    uchar pool_size,  // by now, only pooling size no larger than 3
    uchar pool_stride

)
{
    channel_scal conv_ch_out __attribute__((xcl_data_pack(conv_ch_out)));
    channel_scal pool_final  __attribute__((xcl_data_pack(pool_final))) ;
    DPTYPE line_buf_0[LANE_NUM][POOL_LBUF_DEPTH]   __attribute__((xcl_array_partition(complete,1)));
    DPTYPE line_buf_1[LANE_NUM][POOL_LBUF_DEPTH]   __attribute__((xcl_array_partition(complete,1)));
    uchar  line_buf_ptr;
    uchar  col_pool_cnt;
    uchar  row_pool_cnt;
    uchar  row_cnt;
    DPTYPE row_pool_reg[LANE_NUM] __attribute__((xcl_array_partition(complete,1)));
    DPTYPE col_pool_reg[LANE_NUM] __attribute__((xcl_array_partition(complete,1)));
    DPTYPE pool_reg0[LANE_NUM]  __attribute__((xcl_array_partition(complete,1)));
    DPTYPE pool_reg1[LANE_NUM]  __attribute__((xcl_array_partition(complete,1)));
    DPTYPE pool_reg2[LANE_NUM]  __attribute__((xcl_array_partition(complete,1)));


    line_buf_ptr = 0;
    row_pool_cnt = 0;
    col_pool_cnt = 0;
    for(unsigned int k=0; k<input_num; k++) {

        conv_ch_read_pipe_block(conv_ch_out);

        __attribute__((opencl_unroll_hint))
        for(unsigned char ll=0; ll<LANE_NUM; ll++) {

            if(pool_size==3)
                row_pool_reg[ll] = pool_max(line_buf_1[ll][line_buf_ptr], line_buf_0[ll][line_buf_ptr]);
            else // pool_size==2
                row_pool_reg[ll] = line_buf_0[ll][line_buf_ptr];

            pool_reg0[ll] = pool_max(row_pool_reg[ll], conv_ch_out.lane[ll]);

            if(pool_size==3)
                col_pool_reg[ll] = pool_max(pool_reg1[ll], pool_reg2[ll]);
            else //pool_size==2
                col_pool_reg[ll] = pool_reg1[ll];

            pool_final.lane[ll] = pool_max(col_pool_reg[ll], pool_reg0[ll]);

            // Update line buffer
            line_buf_1[ll][line_buf_ptr] = line_buf_0[ll][line_buf_ptr];
            line_buf_0[ll][line_buf_ptr] = conv_ch_out.lane[ll];

            pool_reg2[ll]=pool_reg1[ll];
            pool_reg1[ll]=pool_reg0[ll];
        }

#ifdef DEBUG_POOL
        printf("Maxpool input_num=%d, line_buf_ptr=%d, row_pool_cnt=%d, col_pool_cnt=%d\n", k, line_buf_ptr, row_pool_cnt, col_pool_cnt);
        printf("        row_cnt=%d\n", row_cnt);
#endif

        // Generates pooling pipeline register wr/rd pointer
        if(row_pool_cnt==(pool_size-1)) {

            if(col_pool_cnt==(pool_size-1)) {
                pool_ch_write_pipe_block(pool_final);
#ifdef DEBUG_POOL
                printf("        reg0=%f, reg1=%f, reg2=%f, max=%f\n", (float)pool_reg[0][0], (float)pool_reg[0][1], (float)pool_reg[0][2], (float)pool_final.lane[0]);
#endif

                col_pool_cnt = (pool_size-pool_stride);
            } else
                col_pool_cnt = col_pool_cnt + 1;
        } else
            col_pool_cnt = 0;

        // Generates line buffer wr/rd pointer
        if(line_buf_ptr==(line_size-1)) {
            line_buf_ptr = 0;

            // Row counters for recognize frames
            if(row_cnt == (line_size-1)) // assuming row_num = line_size, i.e. rectangular frame
                row_cnt = 0;
            else
                row_cnt = row_cnt + 1;

            // Pooling window slide counter for rows
            if(row_cnt == 0)
                row_pool_cnt = 0;
            else if(row_pool_cnt==(pool_size-1))
                row_pool_cnt = (pool_size-pool_stride);
            else
                row_pool_cnt = row_pool_cnt + 1;
        } else {
            line_buf_ptr = line_buf_ptr + 1;
        }

    }
}


// Store Data to Global Memory
__kernel
__attribute__((reqd_work_group_size(1,1,1)))
//__attribute__((xcl_dataflow))
void memWrite(
    // Params Ports
    uchar  out_dim1,
    uchar  out_dim2,
    ushort out_dim3,
    ushort out_dim1xbatch, // out_dim1 x sqrt(batch_size)
    uint   out_dim1x2xbatch, // out_dim1 x out_dim2 x batch_size
    uchar  batch_indx_dim1,
    uchar  batch_indx_dim2,
    uchar  bypass,
    uchar  padd_offset,
    // Data Ports
    __global lane_data * top __attribute__((xcl_data_pack(top)))
)
{
    uchar index_z_item; // max value 256
    ushort index_z_group; // max value 4096

    channel_scal output __attribute__((xcl_data_pack(output)));
    uint local_num = LANE_NUM;
    uint loop_time = (out_dim1*out_dim2*out_dim3);
    __local DPTYPE buffer[LANE_NUM] __attribute__((xcl_array_partition(complete,1)));
    for(unsigned int loop_z=0; loop_z<out_dim3/LANE_NUM; loop_z++) {
        for(unsigned int loop_y=0; loop_y<out_dim2; loop_y++) {
            for(unsigned int loop_x=0; loop_x<out_dim1; loop_x++) {
                for(unsigned int loop=0; loop<LANE_NUM; loop+= VEC_SIZE) {
                    if(loop==0) {
                        if((bypass&0x01)==0x01) {
                            bypass_ch_read_pipe_block(output);
                        } else {
                            pool_ch_read_pipe_block(output);
                        }
                        __attribute__((opencl_unroll_hint))
                        for(uchar ll=0; ll<LANE_NUM; ll++) {
                            buffer[ll]=output.lane[ll];
                        }
                    }
                    index_z_group = (loop_z*LANE_NUM+loop-padd_offset)/VEC_SIZE;
                    index_z_item  = (loop_z*LANE_NUM+loop-padd_offset)%VEC_SIZE;
                    if((loop_z*LANE_NUM+loop-padd_offset)<out_dim3 && (loop_z*LANE_NUM+loop>=padd_offset)) {
                        __attribute__((opencl_unroll_hint))
                        for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
                            top[index_z_group*out_dim1x2xbatch + (loop_y+batch_indx_dim2*out_dim2)*out_dim1xbatch + (loop_x+batch_indx_dim1*out_dim1)].data[vv] = buffer[loop+vv];
                        }
                    }
                }
            }
        }
    }
}

__kernel
__attribute__((xcl_max_work_group_size(LRN_MAX_LOCAL_SIZE)))
void lrn(
    // Params Ports
    uchar data_dim1,
    uchar data_dim2,
    char  frac_dout,
    // Data Ports
    __global lane_data *restrict bottom,
    __global lane_data *restrict top
)
{
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

    __local DPTYPE z_buffer[VEC_SIZE*LRN_MAX_LOCAL_SIZE+LRN_WIN_SIZE]  __attribute__((xcl_array_partition(complete,1))); // allocate two more points for padding
    __local DPTYPE lrn_buffer[VEC_SIZE*LRN_MAX_LOCAL_SIZE]  __attribute__((xcl_array_partition(complete,1)));
    channel_scal data_in;
    channel_scal data_pad_left;
    channel_scal data_pad_right;
    channel_scal data_out;
    lane_data    data_in_partial;
    lane_data    data_left_partial;
    lane_data    data_right_partial;
    lane_data    data_out_partial;
    int          *convert_ptr;
    int          expo;
    uint         manti;
    uint         addr_1, addr_2, addr;
    float        lrn_reg1, lrn_reg2, lrn_tmp, lrn_out;
    short        lrn_cnvt, lrn_cnvt2;
    //Coefficients lookup table for lrn computation
    const float coef0[46] = {9.98312401e-01,8.92383765e-01,8.69534866e-01,8.48001507e-01,8.27672857e-01,8.08269896e-01,7.72814246e-01,7.40785193e-01,7.11686616e-01,6.84743320e-01,6.38046300e-01,5.98139529e-01,5.63585746e-01,5.32842946e-01,4.82570938e-01,4.42066574e-01,4.08721176e-01,3.80120836e-01,3.35733988e-01,3.01782553e-01,2.74896454e-01,2.52503409e-01,2.19044754e-01,1.94367577e-01,1.75328514e-01,1.59766323e-01,1.37073713e-01,1.20695464e-01,1.08253750e-01,9.81965345e-02,8.37272488e-02,7.34111523e-02,6.56398695e-02,5.93964327e-02,5.04776032e-02,4.41593533e-02,3.94211944e-02,3.56262849e-02,3.02252062e-02,2.64117530e-02,2.35583854e-02,2.12767794e-02,1.80355644e-02,1.57509127e-02,1.40434261e-02};
    const float coef1[46] = {-1.07542919e-01,-2.28535953e-02,-2.15331066e-02,-2.03286855e-02,-1.92268508e-02,-3.55023570e-02,-3.20657642e-02,-2.91245494e-02,-2.65861837e-02,-4.68257134e-02,-3.99817597e-02,-3.45887189e-02,-3.02571264e-02,-5.05149626e-02,-4.06040782e-02,-3.34413514e-02,-2.80826706e-02,-4.46757687e-02,-3.40991637e-02,-2.69894342e-02,-2.19616650e-02,-3.37238519e-02,-2.48195600e-02,-1.91265576e-02,-1.52482883e-02,-2.29016249e-02,-1.64847560e-02,-1.25042597e-02,-9.85141038e-03,-1.46114169e-02,-1.03881575e-02,-7.81187564e-03,-6.11526810e-03,-9.00946183e-03,-6.36361270e-03,-4.76376961e-03,-3.71675305e-03,-5.45684726e-03,-3.84135330e-03,-2.86894660e-03,-2.23458481e-03,-3.27498492e-03,-2.30149338e-03,-1.71686994e-03,-1.33609904e-03};
    const float h_inv[46] = {1.22085215e-04,4.88281250e-04,4.88281250e-04,4.88281250e-04,4.88281250e-04,2.44140625e-04,2.44140625e-04,2.44140625e-04,2.44140625e-04,1.22070313e-04,1.22070313e-04,1.22070313e-04,1.22070313e-04,6.10351563e-05,6.10351563e-05,6.10351563e-05,6.10351563e-05,3.05175781e-05,3.05175781e-05,3.05175781e-05,3.05175781e-05,1.52587891e-05,1.52587891e-05,1.52587891e-05,1.52587891e-05,7.62939453e-06,7.62939453e-06,7.62939453e-06,7.62939453e-06,3.81469727e-06,3.81469727e-06,3.81469727e-06,3.81469727e-06,1.90734863e-06,1.90734863e-06,1.90734863e-06,1.90734863e-06,9.53674316e-07,9.53674316e-07,9.53674316e-07,9.53674316e-07,4.76837158e-07,4.76837158e-07,4.76837158e-07,4.76837158e-07};
    const float x_sample[46] = {1.00000000e+00,8.19200000e+03,1.02400000e+04,1.22880000e+04,1.43360000e+04,1.63840000e+04,2.04800000e+04,2.45760000e+04,2.86720000e+04,3.27680000e+04,4.09600000e+04,4.91520000e+04,5.73440000e+04,6.55360000e+04,8.19200000e+04,9.83040000e+04,1.14688000e+05,1.31072000e+05,1.63840000e+05,1.96608000e+05,2.29376000e+05,2.62144000e+05,3.27680000e+05,3.93216000e+05,4.58752000e+05,5.24288000e+05,6.55360000e+05,7.86432000e+05,9.17504000e+05,1.04857600e+06,1.31072000e+06,1.57286400e+06,1.83500800e+06,2.09715200e+06,2.62144000e+06,3.14572800e+06,3.67001600e+06,4.19430400e+06,5.24288000e+06,6.29145600e+06,7.34003200e+06,8.38860800e+06,1.04857600e+07,1.25829120e+07,1.46800640e+07,1.67772160e+07};
    // Load the all data in one line along dim3 into local line buffer
    __attribute__((opencl_unroll_hint))
    for(unsigned char ll=0; ll<VEC_SIZE; ll++) {
        z_buffer[global_z*VEC_SIZE+ll+LRN_WIN_SIZE/2] = bottom[global_z*data_dim2*data_dim1 + global_y*data_dim1+ global_x].data[ll];
    }

    //Padding left
    if(global_z==0) {
        __attribute__((opencl_unroll_hint))
        for(unsigned char ll=0; ll<LRN_WIN_SIZE/2; ll++) {
            z_buffer[ll] = CZERO;
        }
    }

    // Padding right
    if(global_z==(get_global_size(2)-1)) {
        __attribute__((opencl_unroll_hint))
        for(unsigned char ll=0; ll<LRN_WIN_SIZE/2; ll++) {
            z_buffer[VEC_SIZE*get_local_size(2)+ll+LRN_WIN_SIZE/2] = CZERO;
        }
    }

#ifdef DEBUG_LRN
    if(global_z==0&&global_x==0&&global_y==0)
        printf("Kernel LRN: work-item x=%d, y=%d, z=%d(z_local=%d)\n", global_x, global_y, global_z, local_z);
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned char ll=0; ll<VEC_SIZE; ll++) {
        lrn_reg2 = CZERO;
        __attribute__((opencl_unroll_hint))
        for(char k=-LRN_WIN_SIZE/2; k<=LRN_WIN_SIZE/2; k++) {

            lrn_cnvt = z_buffer[global_z*VEC_SIZE+ll+k+LRN_WIN_SIZE/2]<<(-frac_dout);
            lrn_reg1 = convert_float(lrn_cnvt);
            lrn_reg2 += lrn_reg1 * lrn_reg1;
#ifdef DEBUG_LRN
            if(global_z==0&&global_x==0&&global_y==0)
                printf("x=%f(k=%d), ", lrn_reg1, k);
#endif
        }
        convert_ptr = (int*) (&lrn_reg2);
        expo = (EXP_MASK & (*convert_ptr >> MAN_BITS)) - 127;
        manti = ((*convert_ptr) & MAN_MASK);

        addr_1 = ((expo-EXP_STEP_MIN)>>EXP_STEP_LOG)<<MAN_INDEX_BITS;
        addr_2 = (manti>>(MAN_BITS-MAN_INDEX_BITS) & MAN_INDEX_MASK)+1;
        if(expo<EXP_STEP_MIN)
            addr = 0;
        else
            addr = addr_1+addr_2;

        lrn_tmp = ((lrn_reg2-x_sample[addr])*h_inv[addr])*coef1[addr] + coef0[addr];

        lrn_cnvt2 = z_buffer[global_z*VEC_SIZE+ll+LRN_WIN_SIZE/2]<<(-frac_dout);
        lrn_out = lrn_tmp*convert_float(lrn_cnvt2);
        // BUG of Xilinx
        if(lrn_out > 127) {
            lrn_out = 127;
        }

        lrn_buffer[global_z*VEC_SIZE+ll] = convert_char_rte(lrn_out);

#ifdef DEBUG_LRN
        if(global_z==0&&global_x==0&&global_y==0)
            printf("\nKernel LRN (ll=%d): pwlf_x=%f, expo=%d, addr=%d, pwlf_y=%f, lrn=%f\n", ll, lrn_reg2, expo, addr, lrn_tmp, lrn_out);
#endif
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    __attribute__((opencl_unroll_hint))
    for(unsigned char vv=0; vv<VEC_SIZE; vv++) {
        data_out_partial.data[vv]=lrn_buffer[global_z*VEC_SIZE+vv];
    }
    top[global_z*data_dim2*data_dim1 + global_y*data_dim1 + global_x] = data_out_partial;

#ifdef DEBUG_LRN_OUT
    if(global_z==0&&global_x==0&&global_y==0)
        printf("\nKernel LRN OUT: x=%d, y=%d, z=%d, result=%f\n", global_x, global_y, global_z, (float)data_out_partial.data[0]);
#endif

}
