/*
 * ------------------------------------------------------
 *
 *   PipeCNN: An OpenCL-Based FPGA Accelerator for CNNs
 *
 * ------------------------------------------------------
 * Filename:
 *   - hw_param.cl
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
 
#ifndef _HW_PARAM_H
#define _HW_PARAM_H


// Macro architecture parameters
// General
#define VEC_SIZE            8
#define LANE_NUM            16
#define CHN_DEPTH           0
// Conv Kernel
#define PIPE_DEPTH          6
// Pooling Kernel
#define POOL_LBUF_DEPTH     224  // Must be large enough to hold one line (dim1/dim2)
#define POOL_MAX_SIZE       3
// MemWR Kernel
// Note: when VEC_SIZE is larger than LANE_NUM, remember to set the following parameters
#define LOG_LANE_VEC        1     // log2(VEC_SIZE/LANE_NUM)
#define MASK_LANE_VEC       0x01  // LOG_LANE_VEC=1->0x01  2->0x03 3->0x07 ....
// Lrn Kernel
#define LRN_WIN_SIZE        5
#define LRN_MAX_LOCAL_SIZE  (256/VEC_SIZE) // For alexnet the max dim3 size is 256
#define SIGNF_BITS          23   // Floating point format setting
#define ADDR_STEP_LOG       1


#ifdef FORMAT_FP // Using Floating-point format

// Coefficients lookup table for lrn computation
constant float coef0[13] = {1.00000000e+00,9.99985000e-01,9.99940004e-01,9.99760067e-01,9.99041074e-01,9.96177123e-01,9.84910181e-01,9.42656873e-01,8.08499843e-01,5.33567671e-01,2.53198576e-01,9.85382677e-02,3.57579523e-02};
constant float coef1[13] = {-1.49997375e-05,-4.49960628e-05,-1.79937019e-04,-7.18993240e-04,-2.86395109e-03,-1.12669417e-02,-4.22533085e-02,-1.34157029e-01,-2.74932173e-01,-2.80369094e-01,-1.54660309e-01,-6.27803153e-02,-2.30311792e-02};
constant float h_inv[13] = {1.00000000e+00,3.33333333e-01,8.33333333e-02,2.08333333e-02,5.20833333e-03,1.30208333e-03,3.25520833e-04,8.13802083e-05,2.03450521e-05,5.08626302e-06,1.27156576e-06,3.17891439e-07,7.94728597e-08};
constant float x_sample[14] = {0.00000000e+00,1.00000000e+00,4.00000000e+00,1.60000000e+01,6.40000000e+01,2.56000000e+02,1.02400000e+03,4.09600000e+03,1.63840000e+04,6.55360000e+04,2.62144000e+05,1.04857600e+06,4.19430400e+06,1.67772160e+07};

#endif

#endif