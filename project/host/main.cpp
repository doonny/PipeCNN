//////////////////////////////////////////
//
// OpenCL host program template for multiple
// FPGA boards.
//                   
// Created by dongwang@2016.01.10
//
/////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <fstream>

#include <CL/opencl.h>

// user defined library
#include "ocl_util.h"
#include "timer.h"

// CNN network configuration file
#include "../device/hw_param.cl"
#include "layer_config.h"

using namespace std;
using namespace ocl_util;

//----------- Design Parameters --------------//
// select what platform is used
//const char *vendor_name = "Xilinx";
const char *vendor_name = "Altera";
#define DEVICE_TYPE CL_DEVICE_TYPE_ACCELERATOR

//const char *vendor_name = "AMD";
//const char *vendor_name = "Intel";
//#define DEVICE_TYPE CL_DEVICE_TYPE_GPU

// SW System parameters
#define DMA_ALIGNMENT   64
#define MAX_LAYER_NUM   16
#define MAX_BATCH_SIZE  16

#define IN_BUF_SIZE    256*256*64  // Note: the buffer size should be large enough to hold all temperary results
#define OUT_BUF_SIZE   256*256*64
#define FC_BUF_SIZE    32768*MAX_BATCH_SIZE



// AlexNet
// Original problem size
// File size is in num of float numbers
#define IMAGE_FILE_SIZE   (227*227*3)
//#define WEIGHTS_FILE_SIZE 60965224 //fc8-1000
#define WEIGHTS_FILE_SIZE 61063552  //fc8-1024
#define LAYER_NUM         8
#define CONV_NUM          5
const char *weight_file_path = "../data/data_alex/weights.dat";
const char *input_file_path = "../data/data_alex/image.dat";
const char *ref_file_path = "../data/data_alex/fc8.dat";
const char *dump_file_path = "./result_dump.txt";


/*
// VGG16
// Original problem size
// File size is in num of float numbers
#define IMAGE_FILE_SIZE   (224*224*3)
#define WEIGHTS_FILE_SIZE 138455872  //fc8-1024
#define LAYER_NUM         16
#define CONV_NUM          13

const char *weight_file_path = "../data/data_vgg/weights.dat";
const char *input_file_path = "../data/data_vgg/image.dat";
const char *ref_file_path = "../data/data_vgg/fc8.dat";
const char *dump_file_path = "./result_dump.txt";
*/

// Configuration file instructions
enum config_item{
layer_type, // "0" -> conv, "1" -> fc

data_w, data_h, data_n, weight_w, weight_h, weight_n, weight_m, bias_size, //memRd Parameters

memrd_src, //"0"-> data_buf  "1"-> output_buf  "2"->"fc_1_buffer"  "3"->"fc_2_buffer"

conv_x, conv_y, conv_z, conv_stride, conv_padding, conv_split, conv_relu, //Conv Parameters

pool_on, pool_x, pool_y, pool_z, pool_size, pool_stride, // Pooling Parameters

lrn_on,// lrn on/off control

memwr_dst//"0"-> data_buf  "1"-> output_buf  "2"->"fc_1_buffer"  "3"->"fc_2_buffer"

};

enum input_item{

image_w, image_h, image_n, // original image size

batch_size

};

enum output_item{

output_w, output_h, output_n

};


// Define the kernel names used
const char *knl_name_memRd = "memRead";
const char *knl_name_conv  = "coreConv";
const char *knl_name_Pool  = "maxPool";
const char *knl_name_memWr = "memWrite";
const char *knl_name_lrn   = "lrn";


//------------ Global Functions & Variables ------------//
cl_uint num_devices = 0;
cl_platform_id platform_id = NULL;
cl_context context = NULL;
cl_program program = NULL;
scoped_array<cl_device_id> device;
scoped_array<cl_kernel> knl_memRd;
scoped_array<cl_kernel> knl_conv;
scoped_array<cl_kernel> knl_memWr;
scoped_array<cl_kernel> knl_pool;
scoped_array<cl_kernel> knl_lrn;
scoped_array<cl_command_queue> que_memRd;
scoped_array<cl_command_queue> que_conv;
scoped_array<cl_command_queue> que_memWr;
scoped_array<cl_command_queue> que_pool;
scoped_array<cl_mem> data_buf;
scoped_array<cl_mem> output_buf;
scoped_array<cl_mem> weights_buf;
scoped_array<cl_mem> bias_buf;
scoped_array<cl_mem> fc_1_buf;
scoped_array<cl_mem> fc_2_buf;

float *weights;
float *image;
float *data_init;
float *weight_conv[MAX_LAYER_NUM];
float *bias_conv[MAX_LAYER_NUM];
float *output;
float *output_one_item;
float *output_reorder;
float *golden_ref;

int  prepare();
void dumpResult();
void reorderWeights(float *weights, float *buffer, unsigned dim1, unsigned dim2, unsigned dim3, unsigned dim4, unsigned offset, unsigned vecSize, unsigned laneNum);
void reorderBias(float *dataIn, float *bias, unsigned offset, unsigned dim4, unsigned laneNum);
void reorderOutput(float *output, float *output_reorder, unsigned dim1, unsigned dim2, unsigned dim3);
void extractOutput(float *output, float *output_one_item, unsigned item_num, unsigned batch_size, unsigned dim1, unsigned dim2, unsigned dim3);
void cleanup();


int main(int argc, char** argv)
{
	cl_int status;
	float std_err;  // standard errors
	unsigned int err_num;
	
	unsigned int conv_output_num;
	unsigned int conv_loop_cnt;
	unsigned int conv_control;
	unsigned int pool_input_num;
	unsigned int pool_line_size;
	unsigned char pool_bypass;
	unsigned char batch_size_in_dim;
	unsigned char batch_indx_dim1;
	unsigned char batch_indx_dim2;
	
	unsigned int read_buf_size;
	unsigned int batch_item_size;
	
	size_t knl_memRd_global_size[3];
	size_t knl_memRd_local_size[3];
	size_t knl_memWr_global_size[3];
	size_t knl_memWr_local_size[3];
	size_t knl_lrn_global_size[3];
	size_t knl_lrn_local_size[3];
	
	Timer t;  // Timer used for performance measurement
	
	if (argc != 2){
	printf("Error: wrong commad format, usage:\n");  
	printf("%s <binaryfile>\n", argv[0]);
	return EXIT_FAILURE;
	}


	printf("***************************************************\n");
	printf("PipeCNN: An OpenCL-Based FPGA Accelerator for CNNs \n");
	printf("***************************************************\n");
	
	// Prepare compute data
	status = prepare();
	if(status == 1){
		printf("Allocate memory for data and weights failed !!!\n");
		return false;
	}
	
	// Connect to the desired platform
	platform_id = findPlatform(vendor_name);
	if(platform_id == NULL) {
		printf("ERROR: Unable to find the desired OpenCL platform.\n");
		return false;
	}

	// Query the available OpenCL device
	device.reset(getDevices(platform_id, DEVICE_TYPE, &num_devices));
	printf("Platform: %s\n", getPlatformName(platform_id).c_str());
	printf("Using %d device(s)\n", num_devices);
	for(unsigned i = 0; i < num_devices; ++i) {
		printf("  Device %d: %s\n", i, getDeviceName(device[i]).c_str());
		displayDeviceInfo(device[i]);
	}


	// Create the context.
	context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);
	checkError(status, "Failed to create context");

	// Create Program Objects
	char *kernel_file_name=argv[1];

	// Create the program for all device. All devices execute the same kernel.
	program = createProgramFromFile(context, (const char *) kernel_file_name, device, num_devices);
	
	// Create per-device objects.
	que_memRd.reset(num_devices);
	que_conv.reset(num_devices);
	que_memWr.reset(num_devices);
	que_pool.reset(num_devices);
	knl_memRd.reset(num_devices);
	knl_conv.reset(num_devices);
	knl_memWr.reset(num_devices);
	knl_pool.reset(num_devices);
	knl_lrn.reset(num_devices);
	// For each layer a group of buffers are created to store the weights and bias
	weights_buf.reset(num_devices*LAYER_NUM);
	bias_buf.reset(num_devices*LAYER_NUM);
	// Two buffers (data and output) are used as ping-pong buffers for conv layers
	data_buf.reset(num_devices*MAX_BATCH_SIZE);
	output_buf.reset(num_devices*MAX_BATCH_SIZE);
	// Two buffers are used as ping-pong buffers for fc layers
	fc_1_buf.reset(num_devices);
	fc_2_buf.reset(num_devices);
	
	// Create qeues, kernels and mem objs
	for(unsigned i = 0; i < num_devices; ++i) {
		// Command queue
		que_memRd[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
		checkError(status, "Failed to create command queue 0");
		que_conv[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
		checkError(status, "Failed to create command queue 1");
		que_memWr[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
		checkError(status, "Failed to create command queue 2");		
		que_pool[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
		checkError(status, "Failed to create command queue 3");	
		
		// Kernel
		knl_memRd[i] = clCreateKernel(program, knl_name_memRd, &status);
		checkError(status, "Failed to create memRd kernel");

		knl_conv[i] = clCreateKernel(program, knl_name_conv, &status);
		checkError(status, "Failed to create conv kernel");

		knl_pool[i] = clCreateKernel(program, knl_name_Pool, &status);
		checkError(status, "Failed to create pooling kernel");

		knl_memWr[i] = clCreateKernel(program, knl_name_memWr, &status);
		checkError(status, "Failed to create memWr kernel");

		knl_lrn[i] = clCreateKernel(program, knl_name_lrn, &status);
		checkError(status, "Failed to create lrn kernel");

		// Mems
		for(unsigned j = 0; j < LAYER_NUM; ++j){
			// Weights buffers for each layer
			weights_buf[i*LAYER_NUM+j] = clCreateBuffer(context, CL_MEM_READ_ONLY, 
				(layer_config[j][weight_w]*layer_config[j][weight_h]*
				layer_config[j][weight_n]*layer_config[j][weight_m])* sizeof(float), NULL, &status);
			checkError(status, "Failed to create buffer for weights in layer");

			// Bias buffers for each layer
			bias_buf[i*LAYER_NUM+j] = clCreateBuffer(context, CL_MEM_READ_ONLY, 
				layer_config[j][bias_size] * sizeof(float), NULL, &status);
			checkError(status, "Failed to create buffer for bias in layer");
		}
		
		// Create data buffers for each batch item
		for(unsigned j = 0; j < input_config[batch_size]; ++j){
			// Input data buffers
			data_buf[i*input_config[batch_size]+j] = clCreateBuffer(context,  CL_MEM_READ_WRITE, 
				IN_BUF_SIZE * sizeof(float), NULL, &status);
			checkError(status, "Failed to create buffer for data in layer");

			// Output results buffers
			output_buf[i*input_config[batch_size]+j] = clCreateBuffer(context,  CL_MEM_READ_WRITE, 
				OUT_BUF_SIZE * sizeof(float), NULL, &status);
			checkError(status, "Failed to create buffer for output");

			// Load image data into buffers
			status = clEnqueueWriteBuffer(que_memRd[i], data_buf[i*input_config[batch_size]+j], CL_TRUE,
				0, (layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n]) * sizeof(float), data_init, 0, NULL, NULL);
			checkError(status, "Failed to transfer input image");
		}

		// Initializing all weights buffers, blocking write is used
		for(unsigned j = 0; j < LAYER_NUM; ++j){
			status = clEnqueueWriteBuffer(que_memRd[i], weights_buf[i*LAYER_NUM+j], CL_TRUE,
				0, (layer_config[j][weight_w]*layer_config[j][weight_h]*
				layer_config[j][weight_n]*layer_config[j][weight_m])*sizeof(float), weight_conv[j], 0, NULL, NULL);
			checkError(status, "Failed to transfer weight");

			status = clEnqueueWriteBuffer(que_memRd[i], bias_buf[i*LAYER_NUM+j], CL_TRUE,
				0, layer_config[j][bias_size] * sizeof(float), bias_conv[j], 0, NULL, NULL);
			checkError(status, "Failed to transfer bias");
		}
		
		// Allocate fc buffers
		fc_1_buf[i] = clCreateBuffer(context,  CL_MEM_READ_WRITE, 
				FC_BUF_SIZE * sizeof(float), NULL, &status);
		checkError(status, "Failed to create buffer for data in fc layer");

		fc_2_buf[i] = clCreateBuffer(context,  CL_MEM_READ_WRITE, 
				FC_BUF_SIZE * sizeof(float), NULL, &status);
		checkError(status, "Failed to create buffer for data in fc layer");		
	}
	
	// Execute the kernel
	scoped_array<cl_event> memRd_event(num_devices);
	scoped_array<cl_event> conv_event(num_devices);
	scoped_array<cl_event> pool_event(num_devices);
	scoped_array<cl_event> memWr_event(num_devices);
	scoped_array<cl_event> lrn_event(num_devices);
	scoped_array<cl_event> finish_event(num_devices);

	// Recorde the excution time of each operation for each layer
	cl_ulong memWr_time[LAYER_NUM];
	cl_ulong conv_time[LAYER_NUM];
	cl_ulong pool_time[LAYER_NUM];
	cl_ulong memRd_time[LAYER_NUM];
	cl_ulong lrn_time[LAYER_NUM];
	
	// Recorde the start time
	t.start();

	unsigned iter_num;
	// Kernel excutions main loops
	for(unsigned i = 0; i < num_devices; ++i) {
		
		// Each iteration excutes one layer convolution
		// MemRd -> Conv(Relu) -> (MaxPool) -> MemWr -> (Lrn)
		for(unsigned char j = 0; j < LAYER_NUM; ++j){
			
			memWr_time[j] =0;
			conv_time[j]  =0;
			pool_time[j]  =0;
			memRd_time[j] =0;
			lrn_time[j]   =0;
			
			if(j<CONV_NUM)
				iter_num = input_config[batch_size]; // for conv layers, process by batch_size time
			else
				iter_num = 1; // for FC layers, process only one time
			
			// Each iteration process one item in batch
			for(unsigned k = 0; k < iter_num; ++k){
			// Set Arguments
			//
			// Set knl_memRd arguments.
			unsigned argi = 0;
			
			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][data_w]);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
	
			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][data_h]);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
	
			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][weight_w]);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
	
			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][weight_h]);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
	
			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][conv_stride]);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
	
			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][conv_padding]);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
	
			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_uchar), &layer_config[j][conv_split]);
			checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
			
			
			// Select the kernel input mem object source
			// data_buf -> conv1 -> output_buf -> lrn1 -> data_buf -> conv2 -> output_buf -> lrn2 -> data_buf
			// -> conv3 -> output_buf -> conv4 -> output_buf -> ...
			if(layer_config[j][memrd_src]==0){
				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_mem), &data_buf[i*input_config[batch_size]+k]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
			}
			else if(layer_config[j][memrd_src]==1)
			{
				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_mem), &output_buf[i*input_config[batch_size]+k]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
			}
			else if(layer_config[j][memrd_src]==2)
			{
				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_mem), &fc_1_buf[i]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
			}			
			else // 3
			{
				status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_mem), &fc_2_buf[i]);
				checkError(status, "Failed to set argument %d of kernel memRd", argi - 1);
			}
			
			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_mem), &weights_buf[i*LAYER_NUM+j]);
			checkError(status, "Failed to set argument %d kernel memRd", argi - 1);
			
			status = clSetKernelArg(knl_memRd[i], argi++, sizeof(cl_mem), &bias_buf[i*LAYER_NUM+j]);
			checkError(status, "Failed to set argument %d kernel memRd", argi - 1);		
	
			//  Set knl_conv arguments.
			argi = 0;

			conv_output_num = layer_config[j][conv_x]*layer_config[j][conv_y]*layer_config[j][conv_z]/LANE_NUM;
			conv_loop_cnt = layer_config[j][weight_w]*layer_config[j][weight_h]*layer_config[j][weight_n]/VEC_SIZE;
			conv_control = (layer_config[j][conv_relu]&0x01)|(((~layer_config[j][pool_on])&0x01)<<1);

			status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_uint), &conv_output_num);
			checkError(status, "Failed to set argument %d of kernel conv", argi - 1);
	
			status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_uint), &conv_loop_cnt);
			checkError(status, "Failed to set argument %d of kernel conv", argi - 1);
			
			status = clSetKernelArg(knl_conv[i], argi++, sizeof(cl_uint), &conv_control);
			checkError(status, "Failed to set argument %d of kernel conv", argi - 1);
	
			//  Set knl_pool arguments.
			if(layer_config[j][pool_on]){
				argi = 0;
				
				pool_input_num = layer_config[j][conv_x]*layer_config[j][conv_y]*layer_config[j][conv_z]/LANE_NUM;
				pool_line_size = layer_config[j][conv_x];
				status = clSetKernelArg(knl_pool[i], argi++, sizeof(cl_uint), &pool_input_num);
				checkError(status, "Failed to set argument %d of kernel pool", argi - 1);
				
				status = clSetKernelArg(knl_pool[i], argi++, sizeof(cl_uint), &pool_line_size);
				checkError(status, "Failed to set argument %d of kernel pool", argi - 1);
				
				status = clSetKernelArg(knl_pool[i], argi++, sizeof(cl_uint), &layer_config[j][pool_size]);
				checkError(status, "Failed to set argument %d of kernel pool", argi - 1);
				
				status = clSetKernelArg(knl_pool[i], argi++, sizeof(cl_uint), &layer_config[j][pool_stride]);
				checkError(status, "Failed to set argument %d of kernel pool", argi - 1);
			}
	
			//  Set knl_memWr arguments.
			argi = 0;
			unsigned char batch_size_in_dim_log;
			unsigned char mask = 0xff;
			
			pool_bypass = (~layer_config[j][pool_on])&0x01;
			
			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &layer_config[j][pool_x]);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
	
			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &layer_config[j][pool_y]);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
			
			if(j==(CONV_NUM-1)){ // For last Conv Layer, combine all batch data into one fc buffer
				if(input_config[batch_size]==1){
					batch_size_in_dim = 1;
					batch_indx_dim1 = 0;
					batch_indx_dim2 = 0;
				}
				else{
					batch_size_in_dim = log(input_config[batch_size])/log(2);
					batch_size_in_dim_log = log(batch_size_in_dim)/log(2);
					batch_indx_dim1 = k&(~((mask>>batch_size_in_dim_log)<<batch_size_in_dim_log));
					batch_indx_dim2 = k>>batch_size_in_dim_log;
					printf("k=%d (%d, %d)\n", k, batch_size_in_dim, batch_size_in_dim_log);
					printf("batch_indx_dim1=%d\n", batch_indx_dim1);
					printf("batch_indx_dim2=%d\n", batch_indx_dim2);					
				}
			}
			else{ // Normal WR Operations
				batch_size_in_dim = 1;
				batch_indx_dim1 = 0;
				batch_indx_dim2 = 0;				
			}
            
			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &batch_size_in_dim);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
	        
			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &batch_indx_dim1);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
	        
			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &batch_indx_dim2);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);

			status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_uchar), &pool_bypass);
			checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
			
			// Select the kernel output mem object source
			if(layer_config[j][memwr_dst]==0){
				status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_mem), &data_buf[i*input_config[batch_size]+k]);
				checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);		
			}
			else if(layer_config[j][memwr_dst]==1)
			{
				status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_mem), &output_buf[i*input_config[batch_size]+k]);
				checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
			}
			else if(layer_config[j][memwr_dst]==2)
			{
				status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_mem), &fc_1_buf[i]);
				checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
			}
			else // 3
			{
				status = clSetKernelArg(knl_memWr[i], argi++, sizeof(cl_mem), &fc_2_buf[i]);
				checkError(status, "Failed to set argument %d of kernel memWr", argi - 1);
			}

			//  Set knl_lrn arguments.
			if(layer_config[j][lrn_on]){
				argi = 0;
		
				status = clSetKernelArg(knl_lrn[i], argi++, sizeof(cl_uchar), &layer_config[j][pool_x]);
				checkError(status, "Failed to set argument %d of kernel lrn", argi - 1);
		
				status = clSetKernelArg(knl_lrn[i], argi++, sizeof(cl_uchar), &layer_config[j][pool_y]);
				checkError(status, "Failed to set argument %d of kernel lrn", argi - 1);
		
				status = clSetKernelArg(knl_lrn[i], argi++, sizeof(cl_mem), &output_buf[i*input_config[batch_size]+k]);
				checkError(status, "Failed to set argument %d of kernel lrn", argi - 1);	
				
				status = clSetKernelArg(knl_lrn[i], argi++, sizeof(cl_mem), &data_buf[i*input_config[batch_size]+k]);
				checkError(status, "Failed to set argument %d of kernel lrn", argi - 1);	
	
			}
			
			// Excutes Kernel
			//
			if(k == 0)
				printf("\nExecuting Layer %d:\n", j+1);
	
			// kernel memRd
			knl_memRd_global_size[0] = layer_config[j][conv_x]*layer_config[j][weight_w];
			knl_memRd_global_size[1] = layer_config[j][conv_y]*layer_config[j][weight_h];
			knl_memRd_global_size[2] = layer_config[j][conv_z]/LANE_NUM*layer_config[j][weight_n]/VEC_SIZE;
			
			knl_memRd_local_size[0] = layer_config[j][weight_w];
			knl_memRd_local_size[1] = layer_config[j][weight_h];
			knl_memRd_local_size[2] = layer_config[j][weight_n]/VEC_SIZE;

			status = clEnqueueNDRangeKernel(que_memRd[i], knl_memRd[i], 3, NULL, 
									knl_memRd_global_size, knl_memRd_local_size, 0, NULL, &memRd_event[i]);
			checkError(status, "Failed to launch memRd kernel");
			
			if(k == 0)
				printf("\nLaunching kernel MemRd with local size: %d, %d, %d  (global size: %d, %d, %d)\n", 
									(int)knl_memRd_local_size[0], (int)knl_memRd_local_size[1], (int)knl_memRd_local_size[2], 
									(int)knl_memRd_global_size[0], (int)knl_memRd_global_size[1], (int)knl_memRd_global_size[2]);
	
			// kernel conv
			status = clEnqueueTask(que_conv[i], knl_conv[i], 0, NULL, &conv_event[i]);
			checkError(status, "Failed to launch kernel conv kernel");
			if(k == 0)
				printf("\nLaunching single work-item kernel Conv\n");

			// kernel pool
			if(layer_config[j][pool_on]){
				status = clEnqueueTask(que_pool[i], knl_pool[i], 0, NULL, &pool_event[i]);
				checkError(status, "Failed to launch kernel pooling");
				if(k == 0)
					printf("\nLaunching single work-item kernel Pooling\n");
			}

			// kernel memWr
			knl_memWr_global_size[0] = layer_config[j][pool_x];
			knl_memWr_global_size[1] = layer_config[j][pool_y];
			knl_memWr_global_size[2] = layer_config[j][pool_z]/LANE_NUM;
			knl_memWr_local_size[0] = 1;
			knl_memWr_local_size[1] = 1;
			knl_memWr_local_size[2] = 1;

			status = clEnqueueNDRangeKernel(que_memWr[i], knl_memWr[i], 3, NULL, 
									knl_memWr_global_size, knl_memWr_local_size, 0, NULL, &memWr_event[i]);
			checkError(status, "Failed to launch kernel memWr");
			
			if(k == 0)
				printf("\nLaunching kernel MemWr with local size: %d, %d, %d  (global size: %d, %d, %d)\n", 
									(int)knl_memWr_local_size[0], (int)knl_memWr_local_size[1], (int)knl_memWr_local_size[2], 
									(int)knl_memWr_global_size[0], (int)knl_memWr_global_size[1], (int)knl_memWr_global_size[2]);
			
			
			if(layer_config[j][lrn_on]){
				
				knl_lrn_global_size[0] = layer_config[j][pool_x];
				knl_lrn_global_size[1] = layer_config[j][pool_y];
				knl_lrn_global_size[2] = layer_config[j][pool_z]/VEC_SIZE;
				knl_lrn_local_size[0] = 1;
				knl_lrn_local_size[1] = 1;
				knl_lrn_local_size[2] = layer_config[j][pool_z]/VEC_SIZE;
				
				// kernel lrn
				status = clEnqueueNDRangeKernel(que_memWr[i], knl_lrn[i], 3, NULL, 
										knl_lrn_global_size, knl_lrn_local_size, 0, NULL, &lrn_event[i]);
				checkError(status, "Failed to launch kernel lrn");
				
				if(k == 0)
					printf("\nLaunching kernel lrn with local size: %d, %d, %d  (global size: %d, %d, %d)\n", 
										(int)knl_lrn_local_size[0], (int)knl_lrn_local_size[1], (int)knl_lrn_local_size[2], (int)knl_lrn_global_size[0], (int)knl_lrn_global_size[1], (int)knl_lrn_global_size[2]);
			}
			
			// Wait for all kernel to finish
			if(layer_config[j][lrn_on])
				clWaitForEvents(num_devices, lrn_event);
			else
				clWaitForEvents(num_devices, memWr_event);
			
			
			memRd_time[j] += getKernelStartEndTime(memRd_event[i]);
			conv_time[j]  += getKernelStartEndTime(conv_event[i]);
			if(layer_config[j][pool_on])
				pool_time[j] += getKernelStartEndTime(pool_event[i]);
			memWr_time[j] += getKernelStartEndTime(memWr_event[i]);
			if(layer_config[j][lrn_on])
				lrn_time[j] += getKernelStartEndTime(lrn_event[i]);
	
			}// end of batch iteration
			
		}// end of layer iteration
		
	}// end of board iteration



	// Read back the results from the device to verify the output
	// Note：only device0 is used here
	if(num_devices!=1)
		printf("Warnning: only the result from device0 will be verified!!!\n\n");
	
	// Select whith item you would like to compare with the golden ref 
	unsigned batch_item_num = 3;
	if(batch_item_num>(input_config[batch_size]-1)){
		printf("Error: wrong configuration，can't verify the item since it is layer than batch size !!!\n\n");
	}
	
	if(LAYER_NUM<CONV_NUM){ // verify conv results
		read_buf_size = output_config[output_w]*output_config[output_h]*output_config[output_n];
		
	}
	else // verify the last conv and all fc results
		read_buf_size = output_config[output_w]*output_config[output_h]*output_config[output_n]*input_config[batch_size];
	
	// For the last conv layer and all fc layers, read result from one of the fc buffers
	if(layer_config[LAYER_NUM-1][memwr_dst] == 2){
		printf("\nCopyed all batched results from fc_1 buffers.\n");
		status = clEnqueueReadBuffer(que_memWr[0], fc_1_buf[0], CL_FALSE,          // read from device0
			0, sizeof(float) * read_buf_size, (void *)output, 0, NULL, &finish_event[0]);  
		checkError(status, "Failed to set transfer output data");			
	}
	else if(layer_config[LAYER_NUM-1][memwr_dst] == 3){
		printf("\nCopyed all batched results from fc_2 buffers.\n");
		status = clEnqueueReadBuffer(que_memWr[0], fc_2_buf[0], CL_FALSE,          // read from device0
			0, sizeof(float) * read_buf_size, (void *)output, 0, NULL, &finish_event[0]);  
		checkError(status, "Failed to set transfer output data");			
	}
	// For other layers, read results from data and output buffers
	else if(layer_config[LAYER_NUM-1][memwr_dst]^layer_config[LAYER_NUM-1][lrn_on]){// if lrn is used, the mem dst is changed back to src
		printf("\nCopyed one result from NO.%d output buffers.\n", batch_item_num);
		status = clEnqueueReadBuffer(que_memWr[0], output_buf[batch_item_num], CL_FALSE,         // read from device0
			0, sizeof(float) * read_buf_size, (void *)output, 0, NULL, &finish_event[0]);  
		checkError(status, "Failed to set transfer output data");			
	}
	else{
		printf("\nCopyed one results from NO.%d data buffers.\n", batch_item_num);
		status = clEnqueueReadBuffer(que_memWr[0], data_buf[batch_item_num], CL_FALSE,           // read from device0
			0, sizeof(float) * read_buf_size, (void *)output, 0, NULL, &finish_event[0]);  
		checkError(status, "Failed to set transfer output data");
	}

	// Wait for reads to finish
	clWaitForEvents(1, &finish_event[0]);

	//Recorde the end time
	t.stop();
	float time = t.get_time_s();
	printf("\nDone !!!\n\n");
	
	// Evaluate the performance
	// average bandwidth
	//float bw_rd = DATA1_SIZE * sizeof(float) / (time * 1000000.0f);
	//float bw_wr = (DATA1_SIZE+WEIGHT1_SIZE+BIAS1_SIZE) * sizeof(float) / (time * 1000000.0f);
	//float throughput = (DATA1_SIZE*WEIGHT1_WH*WEIGHT1_WH*IMAGE_N) / (time * 1000000000.0f);
	printf("\n-------------------\n");
	printf("\nPerformance Summary\n\n");
	printf("Total runtime: %fs \n\n", time);
	//printf("Average mem read bandwidth: %f MBytes/s\n", bw_rd);
	//printf("Average mem write bandwidth: %f MBytes/s\n", bw_wr);
	//printf("Compute performance: %f GFLOPs/s\n", throughput);

	float kernel_time = 0.0f;
	float batch_float = float(input_config[batch_size]);
	// bandwidth for each device
	printf("Kernel runtime summary:\n");
	for(unsigned j = 0; j < LAYER_NUM; ++j) {
		printf("  Layer-%d:\n", j+1);
		printf("    MemRd: %0.3f ms\n", double(memRd_time[j])/batch_float * 1e-6);
		printf("    Conv : %0.3f ms\n", double(conv_time[j])/batch_float * 1e-6);
		printf("    Pool : %0.3f ms\n", double(pool_time[j])/batch_float * 1e-6);
		printf("    MemWr: %0.3f ms\n", double(memWr_time[j])/batch_float * 1e-6);
		printf("    Lrn  : %0.3f ms\n", double(lrn_time[j])/batch_float * 1e-6);
		kernel_time += conv_time[j];
	}
	printf("\nTotal kernel runtime %0.3f ms \n", double(kernel_time) * 1e-6);
	printf("Batch size = %d, average process time per batch: %0.3f ms \n\n", input_config[batch_size], double(kernel_time/batch_float) * 1e-6);

	// Validate the results
	printf("Start verifying results ...\n");

	
	if(LAYER_NUM>=CONV_NUM){  //Select with batch item you would like to verify from the last conv and all fc output
		printf("Selected item = %d from the combined batch results in fc buffers\n", batch_item_num);
		// Extract one results from all batch arranged results stored in fc buffers
		extractOutput(output, output_one_item, batch_item_num, input_config[batch_size], output_config[output_w], output_config[output_h], output_config[output_n]);
	}
	else
		// Copy results from one of the output/data buffers
		extractOutput(output, output_one_item, 0, 1, layer_config[LAYER_NUM-1][pool_x], layer_config[LAYER_NUM-1][pool_y], layer_config[LAYER_NUM-1][pool_z]);
	
	// Reorder one item of the batch results into scalar format
	reorderOutput(output_one_item, output_reorder, output_config[output_w], output_config[output_h], output_config[output_n]);

	// Compare each results with the golden reference data
	batch_item_size = output_config[output_w]*output_config[output_h]*output_config[output_n];
	err_num = 0;
	for (unsigned int j = 0; j < batch_item_size; j++){
		std_err = pow((output_reorder[j] - golden_ref[j]), 2);
		if(std_err > 1e-3)
			err_num++;
	}
	if(err_num>0)
		printf("Totally %d Wrong Results\n", err_num);
	else
		printf("Check Pass\n");

	// Dump results and golden_ref for debugging
	dumpResult();

	// Release resource
	cleanup();

	return EXIT_SUCCESS;
}

// Read all input data and golden ref data
int prepare()
{
	
	// Load Image data, CNN net weights and golden_results
    ifstream bin_file_r;
    unsigned file_size;
	unsigned weight_size;
	unsigned input_size;
	unsigned output_size;
	unsigned godref_size;
	int ptr=0;
	
	// image and weight files
	weights      = (float *)alignedMalloc(sizeof(float)*WEIGHTS_FILE_SIZE, DMA_ALIGNMENT);
	image        = (float *)alignedMalloc(sizeof(float)*IMAGE_FILE_SIZE, DMA_ALIGNMENT);
	
	// input data
	input_size  = layer_config[0][data_w]*layer_config[0][data_h]*layer_config[0][data_n];
	data_init   = (float *)alignedMalloc(sizeof(float)*input_size, DMA_ALIGNMENT);
	memset(data_init, 0, sizeof(float)*input_size);// fill non-RGB dims with 0

	// final results
	if(LAYER_NUM>=CONV_NUM)// For last conv and all fc layers, all batch results are read back
		output_size = output_config[output_w]*output_config[output_h]*output_config[output_n]*input_config[batch_size];
	else // For other conv layers, only one item of 
		output_size = output_config[output_w]*output_config[output_h]*output_config[output_n];

	godref_size = output_config[output_w]*output_config[output_h]*output_config[output_n];

	output          = (float *)alignedMalloc(sizeof(float)*output_size, DMA_ALIGNMENT); // vectorized results
	output_one_item = (float *)alignedMalloc(sizeof(float)*godref_size, DMA_ALIGNMENT); // one item extracted from batch results
	golden_ref      = (float *)alignedMalloc(sizeof(float)*godref_size, DMA_ALIGNMENT);
    output_reorder  = (float *)alignedMalloc(sizeof(float)*godref_size, DMA_ALIGNMENT); // reordered results for verifying

	if(weights == NULL || image == NULL || golden_ref == NULL || data_init == NULL || output == NULL || output_reorder == NULL)
	{
		printf("Not enough memory !!!");
		alignedFree(weights);
		alignedFree(image);
		alignedFree(data_init);
		alignedFree(golden_ref);
		alignedFree(output_one_item);
		alignedFree(output);
		alignedFree(output_reorder);
		
		return 1;
	}

	// weights and bias	
	for(int j=0; j<LAYER_NUM; j++){
		
		weight_size = (layer_config[j][weight_w]*layer_config[j][weight_h]*layer_config[j][weight_n]*layer_config[j][weight_m]);
		
		weight_conv[j] = (float *)alignedMalloc(sizeof(float)*weight_size, DMA_ALIGNMENT);
		bias_conv[j]   = (float *)alignedMalloc(sizeof(float)*layer_config[j][bias_size], DMA_ALIGNMENT);
		
		memset(weight_conv[j], 0, sizeof(float)*weight_size);// reset with zero
		
		if(weight_conv[j] == NULL || bias_conv[j] == NULL )
		{
			printf("Not enough memory !!!");
			for(int i=0; i<=j; i++){
				alignedFree(weight_conv[i]);
				alignedFree(bias_conv[i]);
			}
			return 1;
		}
	}

    // Weights
    bin_file_r.open(weight_file_path, ios::in | ios::binary);

    if(bin_file_r.is_open())
    {
		//Get file size
		bin_file_r.seekg(0, bin_file_r.end);
		file_size = bin_file_r.tellg();
		bin_file_r.seekg(0, bin_file_r.beg);
		
    	bin_file_r.read((char *)weights, sizeof(float)*WEIGHTS_FILE_SIZE);
    	printf("%d total float weights read \n", file_size/4);
		if(WEIGHTS_FILE_SIZE!=(file_size/4))
			printf("Warning: weight file size does not match user configuration !!!\n");
    	bin_file_r.close();
    }
    else
    	printf("Weights file does not exits !!!\n");

    // Image
    bin_file_r.open(input_file_path, ios::in | ios::binary);

    if(bin_file_r.is_open())
    {
		//Get file size
		bin_file_r.seekg(0, bin_file_r.end);
		file_size = bin_file_r.tellg();
		bin_file_r.seekg(0, bin_file_r.beg);
		
    	bin_file_r.read((char *)image, sizeof(float)*IMAGE_FILE_SIZE);
    	printf("%d bytes image read \n", file_size);
		if(IMAGE_FILE_SIZE!=(file_size/4))
			printf("Warning: image file size does not match user configuration !!!\n");
    	bin_file_r.close();
    }
    else
    	printf("Image file does not exits !!!\n");

    // golden_output
	bin_file_r.open(ref_file_path, ios::in | ios::binary);

    if(bin_file_r.is_open())
    {
		//Get file size
		bin_file_r.seekg(0, bin_file_r.end);
		file_size = bin_file_r.tellg();
		bin_file_r.seekg(0, bin_file_r.beg);
				
    	bin_file_r.read((char *)golden_ref, sizeof(float)*godref_size);
    	printf("%d total output reference read \n\n", file_size/4);
		if(godref_size!=(file_size/4))
			printf("Warning: golden reference file size does not match !!!\n");
    	bin_file_r.close();
    }
    else
    	printf("Golden file does not exits !!!\n");
	
	
	// Copy the image data into the input data buffer for kernel_0
	// The RGB dims are combined into a vector
	// The other dims are filled with 0 (use memset) to fit the vect size of the compute kernel
	for(unsigned i = 0; i<input_config[image_h]; i++){
		for(unsigned j = 0; j<input_config[image_w]; j++){
			for(unsigned k = 0; k<input_config[image_n]; k++){
				data_init[i*input_config[image_w]*VEC_SIZE + j*VEC_SIZE + k] = (float) image[k*input_config[image_w]*input_config[image_h] + i*input_config[image_w] + j];
			}
		}
	}

	// Layer-1
	reorderWeights(weights, weight_conv[0], layer_config[0][weight_w], layer_config[0][weight_h], input_config[image_n], layer_config[0][weight_m], ptr, VEC_SIZE, LANE_NUM);
	ptr+=layer_config[0][weight_w]*layer_config[0][weight_h]*input_config[image_n]*layer_config[0][weight_m];
	reorderBias(weights, bias_conv[0], ptr, layer_config[0][bias_size], LANE_NUM);
	ptr+=layer_config[0][bias_size];
	
	// Other layers
	for(unsigned j=1; j<LAYER_NUM; j++){
		
		if(ptr+layer_config[j][weight_w]*layer_config[j][weight_h]*layer_config[j][weight_n]*layer_config[j][weight_m]>WEIGHTS_FILE_SIZE)
		{
			printf("Error：exceed weight file size !!!\n");
			return 1;
		}
		reorderWeights(weights, weight_conv[j], layer_config[j][weight_w], layer_config[j][weight_h], layer_config[j][weight_n], layer_config[j][weight_m], ptr, VEC_SIZE, LANE_NUM);
		ptr+=layer_config[j][weight_w]*layer_config[j][weight_h]*layer_config[j][weight_n]*layer_config[j][weight_m];
		reorderBias(weights, bias_conv[j], ptr, layer_config[j][bias_size], LANE_NUM);
		ptr+=layer_config[j][bias_size];
	}
	
	return 0;
}


void reorderWeights(float *dataIn, float *weights, unsigned dim1, unsigned dim2, unsigned dim3, unsigned dim4, unsigned offset, unsigned vecSize, unsigned laneNum){

	unsigned k_bound, dim3_dest, n_bound;
	
	if(dim3>vecSize){ // for other Layer
		k_bound = vecSize;
		dim3_dest = dim3;
		n_bound = dim3/vecSize;
	}
	else{ // for Layer 1
		k_bound = dim3;
		dim3_dest = vecSize;
		n_bound = 1;
	}

	for(unsigned m = 0; m<(dim4/laneNum); m++){
		for(unsigned n = 0; n<n_bound; n++){
			for(unsigned i = 0; i<dim2; i++){
				for(unsigned j = 0; j<dim1; j++){
					for(unsigned ll = 0; ll<laneNum; ll++){
						for(unsigned k = 0; k<k_bound; k++){
							//weights[m*dim1*dim2*dim3_dest*laneNum + i*dim1*vecSize*laneNum + j*vecSize*laneNum + ll*vecSize + k] 
							//						= (float) dataIn[offset + (m*laneNum+ll)*k_bound*dim2*dim1 + (n*vecSize+k)*dim1*dim2 + i*dim1 + j];
							weights[m*dim1*dim2*dim3_dest*laneNum + n*dim1*dim2*vecSize*laneNum + i*dim1*vecSize*laneNum + j*vecSize*laneNum + ll*vecSize + k] 
													= (float) dataIn[offset + (m*laneNum+ll)*dim3*dim2*dim1 + (n*vecSize+k)*dim1*dim2 + i*dim1 + j];
						}
					}
				}
			}
		}
	}

}

void reorderBias(float *dataIn, float *bias, unsigned offset, unsigned dim4, unsigned laneNum){
	
	for(unsigned m = 0; m<(dim4/laneNum); m++){
		for(unsigned ll = 0; ll<laneNum; ll++){
			bias[m*laneNum + ll] = (float) dataIn[offset + m*laneNum + ll];
		}
	}
}


// Extract one item from batch results
void extractOutput(float *output, float *output_one_item, unsigned item_num, unsigned batch_size, unsigned dim1, unsigned dim2, unsigned dim3){

	unsigned char mask = 0xff;
	unsigned char batch_size_in_dim;
	unsigned char batch_size_in_dim_log;
	unsigned char batch_indx_dim1;
	unsigned char batch_indx_dim2;
	
	if(batch_size==1){
		batch_size_in_dim = 1;
		batch_indx_dim1 = 0;
		batch_indx_dim2 = 0;
	}
	else{
		batch_size_in_dim = log(batch_size)/log(2);
		batch_size_in_dim_log = log(batch_size_in_dim)/log(2);
		batch_indx_dim1 = item_num&(~((mask>>batch_size_in_dim_log)<<batch_size_in_dim_log));
		batch_indx_dim2 = item_num>>batch_size_in_dim_log;
		printf("Batch Size=%d, verifying NO.%d batch item (indx= %d, %d) ...\n", batch_size, item_num, batch_indx_dim1, batch_indx_dim2);
	}
	
	
	for(unsigned k = 0; k<(dim3/VEC_SIZE); k++){
		for(unsigned i = 0; i<dim2; i++){
			for(unsigned j = 0; j<dim1; j++){
				for(unsigned vv = 0; vv<VEC_SIZE; vv++){
					output_one_item[k*dim2*dim1*VEC_SIZE + i*dim1*VEC_SIZE + j*VEC_SIZE + vv] 
						= output[k*dim2*dim1*batch_size_in_dim*batch_size_in_dim*VEC_SIZE + (i+batch_indx_dim2*dim2)*batch_size_in_dim*dim1*VEC_SIZE + (j+batch_indx_dim1*dim1)*VEC_SIZE + vv];
				}
			}
		}
	}
}


// Re-ordering the vectorized output into scalar form
void reorderOutput(float *output, float *output_reorder, unsigned dim1, unsigned dim2, unsigned dim3){

	for(unsigned i = 0; i<dim2; i++){
		for(unsigned j = 0; j<dim1; j++){
			for(unsigned k = 0; k<(dim3/VEC_SIZE); k++){
				for(unsigned vv = 0; vv<VEC_SIZE; vv++){
					output_reorder[(k*VEC_SIZE+vv)*dim2*dim1 + i*dim1 + j] 
						= output[k*dim2*dim1*VEC_SIZE + i*dim1*VEC_SIZE + j*VEC_SIZE + vv];
				}
			}
		}
	}
}

void dumpResult(){
	 
	ofstream result_file;
	 
	result_file.open(dump_file_path, ios::out);
	 
	for(unsigned i=0; i<output_config[output_n]; i++){
		result_file << "z=" << i << endl;
		for(unsigned j=0; j<output_config[output_h]; j++){
			result_file << "x=" << j << ": ";
			for(unsigned k=0; k<output_config[output_w]; k++){
					result_file << output_reorder[output_config[output_w]*output_config[output_h]*i + output_config[output_w]*j + k] << "(";
					result_file << golden_ref[output_config[output_w]*output_config[output_h]*i + output_config[output_w]*j + k] << ") ";
			}
			result_file << endl;
		}
		result_file << endl;
	}
	result_file.close();
}


// Release all memory resources here
void cleanup()
{

	// Release the opencl runtime resource allocated
	for(unsigned i = 0; i < num_devices; ++i) {
		if(knl_memRd && knl_memRd[i]) {
			clReleaseKernel(knl_memRd[i]);
		}
		if(knl_conv && knl_conv[i]) {
			clReleaseKernel(knl_conv[i]);
		}
		if(knl_memWr && knl_memWr[i]) {
			clReleaseKernel(knl_memWr[i]);
		}		
		if(knl_pool && knl_pool[i]) {
			clReleaseKernel(knl_pool[i]);
		}
		if(knl_lrn && knl_lrn[i]) {
			clReleaseKernel(knl_lrn[i]);
		}
		if(que_memRd && que_memRd[i]) {
			clReleaseCommandQueue(que_memRd[i]);
		}
		if(que_conv && que_conv[i]) {
			clReleaseCommandQueue(que_conv[i]);
		}
		if(que_memWr && que_memWr[i]) {
			clReleaseCommandQueue(que_memWr[i]);
		}
		if(que_pool && que_pool[i]) {
			clReleaseCommandQueue(que_pool[i]);
		}
		if(data_buf && data_buf[i]) {
			clReleaseMemObject(data_buf[i]);
		}
		if(output_buf && output_buf[i]) {
			clReleaseMemObject(output_buf[i]);
		}
		if(weights_buf && weights_buf[i]) {
			clReleaseMemObject(weights_buf[i]);
		}
		if(bias_buf && bias_buf[i]) {
			clReleaseMemObject(bias_buf[i]);
		}
		if(fc_1_buf && fc_1_buf[i]) {
			clReleaseMemObject(fc_1_buf[i]);
		}
		if(fc_2_buf && fc_2_buf[i]) {
			clReleaseMemObject(fc_2_buf[i]);
		}
	}
	
	if(program) {
		clReleaseProgram(program);
	}
	if(context) {
		clReleaseContext(context);
	}

	// Release the memory resource allocated
	alignedFree(weights);
	alignedFree(image);
	alignedFree(data_init);
	for(int j=0; j<LAYER_NUM; j++){
		alignedFree(weight_conv[j]);
		alignedFree(bias_conv[j]);
	}
	alignedFree(golden_ref);
	alignedFree(output);
	alignedFree(output_reorder);
	alignedFree(output_one_item);

}

  
