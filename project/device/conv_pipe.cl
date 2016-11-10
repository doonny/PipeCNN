/*
 * -----------------------------------------------------
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
 *   - v1.0 Initial release
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


#define FORMAT_FP // Using Floating-point format

#include "hw_param.cl"

#pragma OPENCL_EXTENSION cl_altera_channels : enable


// Vectorized data type
typedef struct {
   float data[VEC_SIZE];
} lane_data;

typedef struct {
   lane_data lane[LANE_NUM];
} channel_vec;

typedef struct {
   float lane[LANE_NUM];
} channel_fp32;


channel channel_vec    data_ch    __attribute__((depth(CHN_DEPTH)));
channel channel_vec    weight_ch  __attribute__((depth(CHN_DEPTH)));
channel channel_fp32   bias_ch    __attribute__((depth(CHN_DEPTH)));
channel channel_fp32   conv_ch    __attribute__((depth(CHN_DEPTH)));
channel channel_fp32   pool_ch    __attribute__((depth(CHN_DEPTH)));
channel channel_fp32   bypass_ch  __attribute__((depth(CHN_DEPTH)));


float mac(lane_data input, lane_data weights)
{
	float output = 0.0f;
	
	#pragma unroll
	for(int i=0; i<VEC_SIZE; i++){
		output += input.data[i]*weights.data[i];
	}
	return output;
}


// Fetch Data from Global Memory
__kernel
void memRead(
			// Params Ports
			uchar data_dim1,
			uchar data_dim2,
			uchar weight_dim1,
			uchar weight_dim2,
			uchar stride,
			uchar padding,
			uchar split,
			// Data Ports
			__global const lane_data  *restrict bottom,
			__global channel_vec  *restrict weights,
			__global channel_fp32 *restrict bias        )

{
	int global_x = get_global_id(0);
	int global_y = get_global_id(1);
	int global_z = get_global_id(2);
	
	int local_x = get_local_id(0);
	int local_y = get_local_id(1);
	int local_z = get_local_id(2);
	int block_x = get_group_id(0);
	int block_y = get_group_id(1);
	int block_z = get_group_id(2);
	
	// Input Data, Weights and Bias
	lane_data data_vec;
	channel_vec data_ch_vec;
	channel_vec weight_ch_vec;
	channel_fp32  bias_ch_in;
	unsigned data_offset;
	
	if(split==0)
		data_offset = 0;
	else if(block_z<(get_num_groups(2)>>1))
		data_offset = 0;	
	else
		data_offset = get_local_size(2);	

	
	if(((block_y*stride<padding) && local_y<padding-block_y*stride)||((((get_num_groups(1)-1)-block_y)*stride<padding) && (get_local_size(1)-1-local_y)<padding-((get_num_groups(1)-1)-block_y)*stride)||
		((block_x*stride<padding) && local_x<padding-block_x*stride)||((((get_num_groups(0)-1)-block_x)*stride<padding) && (get_local_size(0)-1-local_x)<padding-((get_num_groups(0)-1)-block_x)*stride)){
		#pragma unroll
		for(unsigned char vv=0; vv<VEC_SIZE; vv++){
			data_vec.data[vv] = 0.0f;
		}
	}
	else
		data_vec = bottom[data_offset*data_dim2*data_dim1 + local_z*data_dim2*data_dim1 + block_y*stride*data_dim1 + (local_y-padding)*data_dim1 + block_x*stride + (local_x-padding)];

	weight_ch_vec = weights[block_z*weight_dim2*weight_dim1*get_local_size(2) + local_z*weight_dim2*weight_dim1 + local_y*weight_dim1 + local_x];
	
	#pragma unroll
	for(unsigned char ll=0; ll<LANE_NUM; ll++){
		data_ch_vec.lane[ll] = data_vec;
	}
	write_channel_altera(data_ch, data_ch_vec);		
	write_channel_altera(weight_ch, weight_ch_vec);		

	if(local_z==0 && local_y==0 && local_x==0){
		bias_ch_in = bias[block_z];
		write_channel_altera(bias_ch, bias_ch_in);

	}
}


__kernel
__attribute__((task))
void coreConv(	
			// Params Ports
			uint  output_num,
			uint  conv_loop_cnt,
			uint  contol //[0]-> relu  [1]->bypass pooling
			)
{
	channel_vec mac_data;
 	channel_vec mac_weight;
	channel_fp32 bias_ch_out;
	channel_fp32 conv_ch_in;
	float bias[LANE_NUM];
	float conv_out[LANE_NUM];
	float lane_accum[LANE_NUM];
	float accum_piped[LANE_NUM][PIPE_DEPTH];

	for(unsigned int k=0; k<output_num; k++){
		
		bias_ch_out = read_channel_altera(bias_ch);

		#pragma unroll
		for(unsigned char ll=0; ll<LANE_NUM; ll++){

			conv_out[ll] = 0.0f;
			bias[ll] = bias_ch_out.lane[ll];

			#pragma unroll
			for(unsigned int p=0; p<PIPE_DEPTH; p++){
				accum_piped[ll][p] = 0.0f;
			}
		}

		for(int j=0; j<conv_loop_cnt; j++){

			mac_data = read_channel_altera(data_ch);
			mac_weight = read_channel_altera(weight_ch);

			#pragma unroll
			for(unsigned char ll=0; ll<LANE_NUM; ll++){
				
				lane_accum[ll] = accum_piped[ll][PIPE_DEPTH-1] + mac(mac_data.lane[ll], mac_weight.lane[ll]);
			
				#pragma unroll
				for(unsigned int p=PIPE_DEPTH-1; p>0; p-- ){
					accum_piped[ll][p]=accum_piped[ll][p-1];
				}
				
				accum_piped[ll][0] = lane_accum[ll];

			}
		}// end of conv loop

		#pragma unroll
		for(unsigned char ll=0; ll<LANE_NUM; ll++){

			#pragma unroll
			for(unsigned i=0; i<PIPE_DEPTH; i++){
				conv_out[ll] += accum_piped[ll][i];
			}

			conv_out[ll] += bias[ll];

			// Relu operation
			if((contol&0x01)==0x01){
				if(conv_out[ll]<=0)
					conv_ch_in.lane[ll] = 0;
				else
					conv_ch_in.lane[ll] = conv_out[ll];
			}
			else
				conv_ch_in.lane[ll] = conv_out[ll];
		}

		// write convoluation results
		if((contol&0x02)==0x02)
			//by-pass pooling
			write_channel_altera(bypass_ch, conv_ch_in);
		else // to pooling kernel
			write_channel_altera(conv_ch, conv_ch_in);

	}// end of output loop
 
}


__kernel
__attribute__((task))
void maxPool(
			// Params Ports
			uint input_num,
			uint line_size,  // line_size should be no larger than POOL_LBUF_DEPTH
			uint pool_size,  // by now, only pooling size no larger than 3
			uint pool_stride
			
			)
{
	channel_fp32 conv_ch_out;
	channel_fp32 pool_final;

	float line_buf_0[LANE_NUM][POOL_LBUF_DEPTH];
	float line_buf_1[LANE_NUM][POOL_LBUF_DEPTH];
	int   line_buf_ptr;
	int   col_pool_cnt;
	int   row_pool_cnt;
	int   row_cnt;
	float row_pool_reg[LANE_NUM];
	float col_pool_reg[LANE_NUM];
	float pool_reg[LANE_NUM][POOL_MAX_SIZE];
	
	line_buf_ptr = 0;
	row_pool_cnt = 0;
	col_pool_cnt = 0;
	for(unsigned int k=0; k<input_num; k++){

		conv_ch_out = read_channel_altera(conv_ch);
	
		#pragma unroll
		for(unsigned char ll=0; ll<LANE_NUM; ll++){
			if(pool_size==3)
				row_pool_reg[ll] = fmax(line_buf_1[ll][line_buf_ptr], line_buf_0[ll][line_buf_ptr]);
			else // pool_size==2
				row_pool_reg[ll] = line_buf_0[ll][line_buf_ptr];
			
			pool_reg[ll][0] = fmax(row_pool_reg[ll], conv_ch_out.lane[ll]);
			
			if(pool_size==3)
				col_pool_reg[ll] = fmax(pool_reg[ll][1], pool_reg[ll][2]);
			else //pool_size==2
				col_pool_reg[ll] = pool_reg[ll][1];

			pool_final.lane[ll] = fmax(col_pool_reg[ll], pool_reg[ll][0]);

			line_buf_1[ll][line_buf_ptr] = line_buf_0[ll][line_buf_ptr];
			line_buf_0[ll][line_buf_ptr] = conv_ch_out.lane[ll];

			#pragma unroll
			for(unsigned int p=POOL_MAX_SIZE-1; p>0; p--){
				pool_reg[ll][p]=pool_reg[ll][p-1];
			}
		}
		
		if(row_pool_cnt==(pool_size-1)){

			if(col_pool_cnt==(pool_size-1)){
				write_channel_altera(pool_ch, pool_final);

				col_pool_cnt = (pool_size-pool_stride);
			}
			else
				col_pool_cnt = col_pool_cnt + 1;
		}
		else
			col_pool_cnt = 0;

		if(line_buf_ptr==(line_size-1)){
			line_buf_ptr = 0;

			if(row_cnt == (line_size-1))
				row_cnt = 0;
			else
				row_cnt = row_cnt + 1;

			if(row_cnt == 0)
				row_pool_cnt = 0;
			else if(row_pool_cnt==(pool_size-1))
				row_pool_cnt = (pool_size-pool_stride);
			else
				row_pool_cnt = row_pool_cnt + 1;
		}
		else{
			line_buf_ptr = line_buf_ptr + 1;
		}

	}
}


// Store Data to Global Memory
__kernel
__attribute__((reqd_work_group_size(1,1,1)))
void memWrite(
				// Params Ports
				uchar out_dim1,
				uchar out_dim2,
				uchar batch_size_in_dim,
				uchar batch_indx_dim1,
				uchar batch_indx_dim2,
				uchar bypass,
				// Data Ports
				#if VEC_SIZE <= LANE_NUM
                __global lane_data *restrict top
				#else
                __global channel_fp32 *restrict top
				#endif
				)
{
	int global_x = get_global_id(0);
	int global_y = get_global_id(1);
	int global_z = get_global_id(2);

	int local_x = get_local_id(0);
	int local_y = get_local_id(1);
	int local_z = get_local_id(2);
	int block_x = get_group_id(0);
	int block_y = get_group_id(1);
	int block_z = get_group_id(2);

	channel_fp32         output;
	#if VEC_SIZE <= LANE_NUM
	lane_data            output_partial;
	#endif
	
	if((bypass&0x01)==0x01)
		output = read_channel_altera(bypass_ch);
	else
		output = read_channel_altera(pool_ch);

	#if VEC_SIZE <= LANE_NUM
		#pragma unroll
		for(unsigned char ll=0; ll<LANE_NUM/VEC_SIZE; ll++){ // need multiple write operations for each work-item to write all LANE_NUM dots
			#pragma unroll
			for(unsigned char vv=0; vv<VEC_SIZE; vv++){
				output_partial.data[vv]=output.lane[ll*VEC_SIZE+vv];
			}
			top[(global_z*(LANE_NUM/VEC_SIZE)+ll)*out_dim2*out_dim1*batch_size_in_dim*batch_size_in_dim + (global_y+batch_indx_dim2*out_dim2)*batch_size_in_dim*out_dim1 + (global_x+batch_indx_dim1*out_dim1)] = output_partial;
		}
	#else //VEC_SIZE > LANE_NUM
		top[(global_z>>LOG_LANE_VEC)*out_dim2*out_dim1*batch_size_in_dim*batch_size_in_dim*VEC_SIZE/LANE_NUM + (global_y+batch_indx_dim2*out_dim2)*batch_size_in_dim*out_dim1*VEC_SIZE/LANE_NUM + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE/LANE_NUM+(global_z&MASK_LANE_VEC)] = output;
	#endif
}


__kernel
__attribute__((max_work_group_size(LRN_MAX_LOCAL_SIZE)))
void lrn(
			// Params Ports
			uchar data_dim1,
			uchar data_dim2,
			// Data Ports
			__global lane_data *restrict bottom,
			__global lane_data *restrict top
		)
{
	int global_x = get_global_id(0);
	int global_y = get_global_id(1);
	int global_z = get_global_id(2);

	int local_x = get_local_id(0);
	int local_y = get_local_id(1);
	int local_z = get_local_id(2);
	int block_x = get_group_id(0);
	int block_y = get_group_id(1);
	int block_z = get_group_id(2);

	__local float z_buffer[VEC_SIZE*LRN_MAX_LOCAL_SIZE+LRN_WIN_SIZE]; // allocate two more points for padding
	__local float lrn_buffer[VEC_SIZE*LRN_MAX_LOCAL_SIZE];
	channel_fp32 data_in;
	channel_fp32 data_pad_left;
	channel_fp32 data_pad_right;
	channel_fp32 data_out;
	lane_data    data_in_partial;
	lane_data    data_left_partial;
	lane_data    data_right_partial;
	lane_data    data_out_partial;
	int          *expo_pt;
	int          expo;
	float        lrn_reg1, lrn_reg2, lrn_tmp, lrn_out;
	unsigned int addr;

	#pragma unroll
	for(unsigned char ll=0; ll<VEC_SIZE; ll++){
		z_buffer[global_z*VEC_SIZE+ll+LRN_WIN_SIZE/2] = bottom[global_z*data_dim2*data_dim1 + global_y*data_dim1+ global_x].data[ll];
	}
	
	if(global_z==0){
		#pragma unroll
		for(unsigned char ll=0; ll<LRN_WIN_SIZE/2; ll++){
			z_buffer[ll] = 0.0f;
		}
	}

	if(global_z==(get_global_size(2)-1)){
		#pragma unroll
		for(unsigned char ll=0; ll<LRN_WIN_SIZE/2; ll++){
			z_buffer[VEC_SIZE*get_local_size(2)+ll+LRN_WIN_SIZE/2] = 0.0f;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for(unsigned char ll=0; ll<VEC_SIZE; ll++){
		lrn_reg2 = 0.0f;
		#pragma unroll
		for(char k=-LRN_WIN_SIZE/2; k<=LRN_WIN_SIZE/2; k++){
			lrn_reg1 = z_buffer[global_z*VEC_SIZE+ll+k+LRN_WIN_SIZE/2];
			lrn_reg2 += lrn_reg1 * lrn_reg1;
		}
		expo_pt = (int*) (&lrn_reg2);
		expo = (0xff & (*expo_pt >> SIGNF_BITS)) - 127;
		if(expo<0) expo = 0; // if x<1, use the first segment
		addr = (expo >> ADDR_STEP_LOG) + 1;

		lrn_tmp = ((lrn_reg2-x_sample[addr])*h_inv[addr])*coef1[addr] + coef0[addr];	
		
		lrn_out = lrn_tmp*z_buffer[global_z*VEC_SIZE+ll+LRN_WIN_SIZE/2];

		lrn_buffer[global_z*VEC_SIZE+ll] = lrn_out;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	#pragma unroll
	for(unsigned char vv=0; vv<VEC_SIZE; vv++){
		data_out_partial.data[vv]=lrn_buffer[global_z*VEC_SIZE+vv];
	}
	top[global_z*data_dim2*data_dim1 + global_y*data_dim1 + global_x] = data_out_partial;

}

