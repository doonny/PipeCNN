# PipeCNN

## About 
**PipeCNN** is an OpenCL-based FPGA Accelerator for Large-Scale Convolutinal Neural Networks (CNNs).
There is a growing trend among the FPGA community to utilize High Level Synthesis (HLS) tools to design
and implement customized circuits on FPGAs. Compared with RTL-based design methodology, the HLS tools provide faster hardware development
cycle by automatically synthesizing an algorithm in high-level languages (e.g. C/C++) to RTL/hardware. [OpenCL™](https://www.khronos.org/opencl/) is an open, emergying cross-platform parallel programming language that can be used in both GPU and FPGA developments. The main goal of this project is to provide a generic, yet efficient OpenCL-based design of CNN accelerator on FPGAs. Our design is scalable both in performance and hardware resource, and thus can be deployed on a variety of FPGA platforms.

## How to Use

First, download the pre-trained CNN models, input test vectors and golden reference files from PipeCNN's own [ModelZoo](https://github.com/doonny/PipeCNN/tree/master/data). Place the data in the correct folder. Compile the project by using the Makefile provided. After finishing the compilation, simply type the following command to run PipeCNN:
```
./run.exe conv.aocx
```
For users who are using Xilinx's SDx environments, it is recommended to use the IDE instead of makefiles.
For more detailed user instructions, please refer to the [User Instructions](https://github.com/doonny/PipeCNN/tree/master/documents).

## Boards and Performances
Currently, we use [Intel's OpenCL SDK](https://www.altera.com/products/design-software/embedded-software-developers/opencl/overview.html) v16.1 toolset for compilation of the OpenCL code and implementation of the generated RTL on Altera's FPGAs. For Xilinx FPGAs, the [SDAccel](https://www.xilinx.com/products/design-tools/software-zone/sdaccel.html) and [SDSoc](https://www.xilinx.com/products/design-tools/software-zone/sdsoc.html) development environments v2017.2 can be used. PipeCNN has been tested and evaluated on the following FPGA boards/platforms. We welcome other vendors/researches to provide performance and cost information on other FPGA platforms/boards.


* Terasic's [DE5-net](http://www.terasic.com.cn/cgi-bin/page/archive.pl?Language=China&CategoryNo=179&No=727) (Stratix-V A7 FPGA)
* Terasic's [DE5a-net](http://www.terasic.com.cn/cgi-bin/page/archive.pl?Language=China&CategoryNo=251&No=988) (Arria-10 1150 FPGA)
* Terasic's [DE1-soc](http://www.terasic.com.cn/cgi-bin/page/archive.pl?Language=China&CategoryNo=180&No=870) (Cyclone-V SEA5 FPGA)
* Terasic's [DE10-standard](http://www.terasic.com.cn/cgi-bin/page/archive.pl?Language=China&CategoryNo=180&No=1105) (Cyclone-V SXC6 FPGA)
* Xilinx's [KCU1500](https://www.xilinx.com/products/boards-and-kits/dk-u1-kcu1500-g.html) (XCKU115 FPGA)


## Update Plans
* Implementation of Faster-RCNN (TBD.)
* Optimization for DE5a-net (Arria-10) targeting 500 fps of AlexNet (end of August)
* Support for sparse or Winograd-based convolution algorithms

## Citation
Please kindly cite our work of PipeCNN if it helps your research:
```
Dong Wang, Jiangjing An and Ke Xu, “PipeCNN: An OpenCL-Based FPGA Accelerator for Large-Scale Convolution Neuron Networks”, https://arxiv.org/abs/1611.02450, 2016.
```



## Related Works
There are other FPGA accelerators that also adopt HLS-based design scheme. Some brilliant works are listed as follow. Note that PipeCNN is the first, and only one that is Open-Source （￣︶￣）↗
* U. Aydonat, S. O'Connell, D. Capalija, A. C. Ling, and G. R. Chiu. "An OpenCL™ Deep Learning Accelerator on Arria 10," *in Proc. FPGA 2017*.
* N. Suda, V. Chandra, G. Dasika, A. Mohanty, Y. F. Ma, S. Vrudhula, J. S. Seo, and Y. Cao, "Throughput-Optimized OpenCL-based FPGA accelerator for large-scale convolutional neural networks," *in Proc. FPGA 2016*.
* C. Zhang, P. Li, G. Sun, Y. Guan, B. J. Xiao, and J. Cong, "Optimizing FPGA-based accelerator design for deep convolutional neural networks," *in Proc. FPGA 2015*.
