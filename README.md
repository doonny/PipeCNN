# PipeCNN

## About 
**PipeCNN** is an OpenCL-based FPGA Accelerator for Large-Scale Convolutinal Neural Networks (CNNs).
There is a growing trend among the FPGA community to utilize High Level Synthesis (HLS) tools to design
and implement customized circuits on FPGAs. Compared with RTL-based design methodology, the HLS tools provide faster hardware development
cycle by automatically synthesizing an algorithm in high-level languages (e.g. C/C++) to RTL/hardware. [OpenCL](https://www.khronos.org/opencl/) is an open, emergying cross-platform parallel programming language that can be used in both GPU and FPGA developments. This project provides an efficient OpenCL-based design of CNN accelerator on FPGAs. Our design is scalable in both performance and hardware resource, and thus can be implemented on a variety of FPGA platforms. Currently, we only use the [Altera OpenCL SDK](https://www.altera.com/products/design-software/embedded-software-developers/opencl/overview.html) toolset for compilation of the OpenCL code and implementation of the generated RTL on Altera's FPGAs. The final designs have been tested on Terasic's [DE5-net](https://www.altera.com.cn/solutions/partners/partner-profile/terasic-inc-/board/de5-net-fpga-development-kit.html) board.



## User Instructions
Before starting to use this project, you need to install Altera's OpenCL SDK toolset on a Linux or Windows desktop computer, on which a supported FPGA board is correctly installed. You also need to prepare several binary files which contain test data (trained nueral network models and image data) to run image classification (The files are too large to be included here). Integration with Caffe and OpenCV remains to be done. However, there is one usefull tool that can be used to prepare these files:
```
https://github.com/pmgysel/alexnet-forwardpath
```

After all things are ready, run the makefile included in this project, and it will take around one hour to finish all the compilations. Then, there will be two files generated as
* *run.exe*
* *conv.aocx*

Simply start the accelerator by typing
```
./run.exe conv.aocx
```

The results will be like this:
```
*************************************************
PipeCNN: An OpenCL-Based FPGA Accelerator for CNNs 
*************************************************
61063552 total float weights read 
618348 bytes image read 
1024 total output reference read 

Platform: Altera SDK for OpenCL
Using 1 device(s)
  Device 0: de5net_a7 : Altera's Preferred Board
Device OpenCL Version: OpenCL 1.0 Altera SDK for OpenCL, Version 15.1
Device Max Compute Units: 1
Device Max WorkGroup Size: 2147483647
Device Max WorkItem Size: 2147483647
Device Global Memory Size: 4096 MBytes
Device Local Memory Size: 16 KBytes
Device Max Clock Freq: 1000 Mhz

Loading kernel/binary from file conv.aocx

Executing Layer 1:

Launching kernel MemRd with local size: 11, 11, 1  (global size: 605, 605, 6)

Launching single work-item kernel Conv

Launching single work-item kernel Pooling

Launching kernel MemWr with local size: 1, 1, 1  (global size: 27, 27, 6)

Launching kernel lrn with local size: 1, 1, 12  (global size: 27, 27, 12)

Executing Layer 2:

Launching kernel MemRd with local size: 5, 5, 6  (global size: 135, 135, 96)

Launching single work-item kernel Conv

Launching single work-item kernel Pooling

Launching kernel MemWr with local size: 1, 1, 1  (global size: 13, 13, 16)

Launching kernel lrn with local size: 1, 1, 32  (global size: 13, 13, 32)

Executing Layer 3:

Launching kernel MemRd with local size: 3, 3, 32  (global size: 39, 39, 768)

Launching single work-item kernel Conv

Launching kernel MemWr with local size: 1, 1, 1  (global size: 13, 13, 24)

Executing Layer 4:

Launching kernel MemRd with local size: 3, 3, 24  (global size: 39, 39, 576)

Launching single work-item kernel Conv

Launching kernel MemWr with local size: 1, 1, 1  (global size: 13, 13, 24)

Executing Layer 5:

Launching kernel MemRd with local size: 3, 3, 24  (global size: 39, 39, 384)

Launching single work-item kernel Conv

Launching single work-item kernel Pooling

Launching kernel MemWr with local size: 1, 1, 1  (global size: 6, 6, 16)

Executing Layer 6:

Launching kernel MemRd with local size: 6, 6, 32  (global size: 24, 24, 8192)

Launching single work-item kernel Conv

Launching kernel MemWr with local size: 1, 1, 1  (global size: 4, 4, 256)

Executing Layer 7:

Launching kernel MemRd with local size: 1, 1, 512  (global size: 4, 4, 131072)

Launching single work-item kernel Conv

Launching kernel MemWr with local size: 1, 1, 1  (global size: 4, 4, 256)

Executing Layer 8:

Launching kernel MemRd with local size: 1, 1, 512  (global size: 4, 4, 32768)

Launching single work-item kernel Conv

Launching kernel MemWr with local size: 1, 1, 1  (global size: 4, 4, 64)

Copyed all batched results from fc_2 buffers.

Done !!!


-------------------

Performance Summary

Total runtime: 0.857099s 

Kernel runtime summary:
  Layer-1:
    MemRd: 12.348 ms
    Conv : 12.325 ms
    Pool : 12.302 ms
    MemWr: 12.282 ms
    Lrn  : 0.893 ms
  Layer-2:
    MemRd: 9.919 ms
    Conv : 9.881 ms
    Pool : 9.840 ms
    MemWr: 9.805 ms
    Lrn  : 0.300 ms
  Layer-3:
    MemRd: 6.866 ms
    Conv : 6.809 ms
    Pool : 0.000 ms
    MemWr: 6.746 ms
    Lrn  : 0.000 ms
  Layer-4:
    MemRd: 5.202 ms
    Conv : 5.128 ms
    Pool : 0.000 ms
    MemWr: 5.052 ms
    Lrn  : 0.000 ms
  Layer-5:
    MemRd: 3.532 ms
    Conv : 3.450 ms
    Pool : 3.364 ms
    MemWr: 3.293 ms
    Lrn  : 0.000 ms
  Layer-6:
    MemRd: 4.842 ms
    Conv : 4.837 ms
    Pool : 0.000 ms
    MemWr: 4.831 ms
    Lrn  : 0.000 ms
  Layer-7:
    MemRd: 1.096 ms
    Conv : 1.090 ms
    Pool : 0.000 ms
    MemWr: 1.084 ms
    Lrn  : 0.000 ms
  Layer-8:
    MemRd: 0.282 ms
    Conv : 0.276 ms
    Pool : 0.000 ms
    MemWr: 0.266 ms
    Lrn  : 0.000 ms

Total kernel runtime 700.753 ms 
Batch size = 16, average process time per batch: 43.797 ms 

Start verifying results ...
Selected item = 3 from the combined batch results in fc buffers
Batch Size=16, verifying NO.3 batch item (indx= 3, 0) ...
Check Pass

```
Note that current host code only read one image file which is reused for each batch process.


## Configurations
Configuration of a new implementation with different performance and hardware resource utilizations is controlled by a header file located in *device/hw_param.cl*. Change the following macros
* VEC_SIZE
* LANE_NUM

to appropriate ones. The default setting is VEC_SIZE=8, LANE_NUM=16, which achieves the shortest classification time on the DE5-net board.

Configuration of different CNN models is done by a header file located in *host/layer_config.h*. Select one of the model configurations provided and recompile the host before running the test. Currently, the following models have been tested:
* AlexNet (CaffeNet)
* Vgg-16


## Update Plans
* fixed-point (8/16-bit) implementation with at least 4x performance improvements (around May-15)
* optimizations for DE1-soc (Cyclone-V) targeting 20 fps of AlexNet (around end of June)
* implementation of Faster-RCNN (end of August)
* optimizations for DE5a-net (Arria-10) targeting 500 fps of AlexNet (end of July)


## Citation
Please kindly cite our work of PipeCNN if it helps your research:
```
Dong Wang, Jiangjing An and Ke Xu, “PipeCNN: An OpenCL-Based FPGA Accelerator for Large-Scale Convolution Neuron Networks”, https://arxiv.org/abs/1611.02450, 2016.
```



## Reference
There are other FPGA accelerators that also adopt HLS-based design scheme. Some brilliant works are listed as follow. Note that PipeCNN is the first Open-Source one （￣︶￣）↗
* N. Suda, V. Chandra, G. Dasika, A. Mohanty, Y. F. Ma, S. Vrudhula, J. S. Seo, and Y. Cao, "Throughput-Optimized OpenCL-based FPGA accelerator for large-scale convolutional neural networks," *in Proc. FPGA 2016*.
* C. Zhang, P. Li, G. Sun, Y. Guan, B. J. Xiao, and J. Cong, "Optimizing FPGA-based accelerator design for deep convolutional neural networks," *in Proc. FPGA 2015*.



##

There are still many things that can be done to futher optimize the performance and resource utilization. We welcome students, researchers to join our Lab or directly contribute to this project. 
