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

#ifndef _HW_PARAM_H
#define _HW_PARAM_H


// Macro architecture parameters
// General
#define VEC_SIZE            4              // larger than 4, i.e., 4, 8, 16, ...
#define LANE_NUM            16             // larger than 1, for alexnet: 2, 3, 4, 8, 12, 15, 16, 22, 28, 32, 34, 48, 50, 51, 52, 64, ...
#define CHN_DEPTH           0
//MemRD Kernel
#define CONV_GP_SIZE_X      7
#define CONV_GP_SIZE_Y      1              // In this version, CONV_GP_SIZE_Y must be 1
#define WIN_BUF_SIZE        9216/VEC_SIZE  // for AlexNet  batch=1
#define WEIGHT_BUF_SIZE     9216/VEC_SIZE  // for AlexNet  batch=1
//#define WIN_BUF_SIZE        25088/VEC_SIZE // for VGG-16  batch=1
//#define WEIGHT_BUF_SIZE     25088/VEC_SIZE // for VGG-16  batch=1
//#define WIN_BUF_SIZE        CONV_GP_SIZE_X*9216/VEC_SIZE  // for AlexNet  batch>=4
//#define WEIGHT_BUF_SIZE     9216/VEC_SIZE                 // for AlexNet  batch>=4
// Conv Kernel
#define PIPE_DEPTH          6
// Pooling Kernel
#define POOL_LBUF_DEPTH     224            // Must be large enough to hold one line (dim1/dim2)
#define POOL_MAX_SIZE       3
// Lrn Kernel
#define LRN_WIN_SIZE        5
#define LRN_MAX_LOCAL_SIZE  (256/VEC_SIZE) // For alexnet the max dim3 size is 256
#define MAN_BITS            23             // Floating point format setting
#define EXP_MASK            0xFF           // Floating point format setting
#define MAN_MASK            0x7FFFFF       // Floating point format setting
#define EXP_STEP_MIN        13             // PWLF table setting
#define EXP_STEP_LOG        0              // PWLF table setting
#define MAN_INDEX_BITS      2              // PWLF table setting
#define MAN_INDEX_MASK      0x03           // PWLF table setting
// Parameters for fixed-point design
#define CZERO       0x00     // constant zero
#define MASK8B      0xFF     // used for final rounding
#define MASK9B      0x1FE    // used for final rounding
#define MASKSIGN    0x80     // used for final rounding
// #define MASK_ACCUM  0x1FFFFFFF // not used for reducing mac pipeline logic cost (when PIPE_DEPTH=6, MASK_ACCUM has 16+13=29 bits)
//#define MASK_MULT   0x1FFFFF   // not used for reducing mac pipeline logic cost (four parallel mac, max VEC_SIZE is 32, MASK_MULT has 16+5=21 bits)
#define MASK_ACCUM  0xFFFFFFFF // use this value
#define MASK_MULT   0xFFFFFFFF // use this value


#ifdef USE_ROM
//Coefficients lookup table for lrn computation
constant float coef0[46] = {9.98312401e-01,8.92383765e-01,8.69534866e-01,8.48001507e-01,8.27672857e-01,8.08269896e-01,7.72814246e-01,7.40785193e-01,7.11686616e-01,6.84743320e-01,6.38046300e-01,5.98139529e-01,5.63585746e-01,5.32842946e-01,4.82570938e-01,4.42066574e-01,4.08721176e-01,3.80120836e-01,3.35733988e-01,3.01782553e-01,2.74896454e-01,2.52503409e-01,2.19044754e-01,1.94367577e-01,1.75328514e-01,1.59766323e-01,1.37073713e-01,1.20695464e-01,1.08253750e-01,9.81965345e-02,8.37272488e-02,7.34111523e-02,6.56398695e-02,5.93964327e-02,5.04776032e-02,4.41593533e-02,3.94211944e-02,3.56262849e-02,3.02252062e-02,2.64117530e-02,2.35583854e-02,2.12767794e-02,1.80355644e-02,1.57509127e-02,1.40434261e-02};
constant float coef1[46] = {-1.07542919e-01,-2.28535953e-02,-2.15331066e-02,-2.03286855e-02,-1.92268508e-02,-3.55023570e-02,-3.20657642e-02,-2.91245494e-02,-2.65861837e-02,-4.68257134e-02,-3.99817597e-02,-3.45887189e-02,-3.02571264e-02,-5.05149626e-02,-4.06040782e-02,-3.34413514e-02,-2.80826706e-02,-4.46757687e-02,-3.40991637e-02,-2.69894342e-02,-2.19616650e-02,-3.37238519e-02,-2.48195600e-02,-1.91265576e-02,-1.52482883e-02,-2.29016249e-02,-1.64847560e-02,-1.25042597e-02,-9.85141038e-03,-1.46114169e-02,-1.03881575e-02,-7.81187564e-03,-6.11526810e-03,-9.00946183e-03,-6.36361270e-03,-4.76376961e-03,-3.71675305e-03,-5.45684726e-03,-3.84135330e-03,-2.86894660e-03,-2.23458481e-03,-3.27498492e-03,-2.30149338e-03,-1.71686994e-03,-1.33609904e-03};
constant float h_inv[46] = {1.22085215e-04,4.88281250e-04,4.88281250e-04,4.88281250e-04,4.88281250e-04,2.44140625e-04,2.44140625e-04,2.44140625e-04,2.44140625e-04,1.22070313e-04,1.22070313e-04,1.22070313e-04,1.22070313e-04,6.10351563e-05,6.10351563e-05,6.10351563e-05,6.10351563e-05,3.05175781e-05,3.05175781e-05,3.05175781e-05,3.05175781e-05,1.52587891e-05,1.52587891e-05,1.52587891e-05,1.52587891e-05,7.62939453e-06,7.62939453e-06,7.62939453e-06,7.62939453e-06,3.81469727e-06,3.81469727e-06,3.81469727e-06,3.81469727e-06,1.90734863e-06,1.90734863e-06,1.90734863e-06,1.90734863e-06,9.53674316e-07,9.53674316e-07,9.53674316e-07,9.53674316e-07,4.76837158e-07,4.76837158e-07,4.76837158e-07,4.76837158e-07};
constant float x_sample[46] = {1.00000000e+00,8.19200000e+03,1.02400000e+04,1.22880000e+04,1.43360000e+04,1.63840000e+04,2.04800000e+04,2.45760000e+04,2.86720000e+04,3.27680000e+04,4.09600000e+04,4.91520000e+04,5.73440000e+04,6.55360000e+04,8.19200000e+04,9.83040000e+04,1.14688000e+05,1.31072000e+05,1.63840000e+05,1.96608000e+05,2.29376000e+05,2.62144000e+05,3.27680000e+05,3.93216000e+05,4.58752000e+05,5.24288000e+05,6.55360000e+05,7.86432000e+05,9.17504000e+05,1.04857600e+06,1.31072000e+06,1.57286400e+06,1.83500800e+06,2.09715200e+06,2.62144000e+06,3.14572800e+06,3.67001600e+06,4.19430400e+06,5.24288000e+06,6.29145600e+06,7.34003200e+06,8.38860800e+06,1.04857600e+07,1.25829120e+07,1.46800640e+07,1.67772160e+07};
#endif

#endif