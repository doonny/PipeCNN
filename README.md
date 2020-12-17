# PipeCNN

## About 
**PipeCNN** is an OpenCL-based FPGA Accelerator for Large-Scale Convolutional Neural Networks (CNNs).
There is a growing trend among the FPGA community to utilize High Level Synthesis (HLS) tools to design
and implement customized circuits on FPGAs. Compared with RTL-based design methodology, the HLS tools provide faster hardware development
cycle by automatically synthesizing an algorithm in high-level languages (e.g. C/C++) to RTL/hardware. [OpenCL™](https://www.khronos.org/opencl/) is an open, emergying cross-platform parallel programming language that can be used in both GPU and FPGA developments. The main goal of this project is to provide a generic, yet efficient OpenCL-based design of CNN accelerator on FPGAs. PipeCNN utilizes ***Pipe**lined **CNN*** functional kernels to achieved improved throughput in inference computation. Our design is scalable both in performance and hardware resource, and thus can be deployed on a variety of FPGA platforms. PipeCNN supports both Intel OpenCL SDK and Xilinx Vitis based FPGA design flow.

## How to Use

First, download the pre-trained CNN models, input test vectors and golden reference files from PipeCNN's own ModelZoo (instructions are located in the "data" folder inside each project folder). Place the data in the correct folder. Then, compile the project by using the Makefile provided. After finishing the compilation, simply type the following command to run PipeCNN:
```
./run.exe conv.aocx
```
The ModelZoo now provides pre-quantized model for the following networks:
* VGG-16
* ResNet-50

For more detailed instructions, please check out the [User Instructions](https://github.com/doonny/PipeCNN/tree/master/documents).

## Supported Tools
Currently, we are using [Intel's OpenCL SDK](https://www.intel.com/content/www/us/en/software/programmable/sdk-for-opencl/overview.html) and [Xilinx Vitis](https://china.xilinx.com/products/design-tools/vitis/vitis-platform.html) tool kit to compile of the OpenCL/HLS code and implementate of the generated RTL on FPGAs. 

* Intel OpenCL SDK Pro v20.1
* Xilinx Vitis 2020.1

## Tested Boards
The following boards have been tested working:
* Terasic's DE5a-net-ddr4 (Arria-10 GX1150 FPGA)
* Intel's Arria-10 Dev Kit (Arria-10 GX1150 FPGA)
* Xilinx's U50 Acceleration Card (VU35P FPGA)
* Xilinx's ZCU102 Dev Board (ZU9EG FPGA)
* Xilinx's ZC706 Dev Board (Zynq-7045 FPGA)

PipeCNN may also run on other FPGA boards, which includes Terasic's DE10-standard/DE10-nano, Intel's PAC cards, Xilinx Ultra96-v2 boards. However, due to limited time and resourse, we have not verified that yet. Please let us know if you would like to share your results on other FPGA boards.

## Demos
Now you can run classification on the ImageNet dataset by using PipeCNN, and measure the top-1/5 accuracy for different CNN models.

To run this demo, first, set **USE_OPENCV = 1** in the Makefile. Secondly, download the ImageNet validation dataset, extract and place all the pictures in the "/data" folder. Rename the variable "picture_file_path_head" in the host file to indicate the correct image data set path. Finally, recompile the host program and run PipeCNN.

The following piture shows that the demo runs on our own computer with the DE5-net board.

![DE5-net-Demo](documents/Demo-DE5-net.gif)

## Performances
It's been four years since the release of the this project. Deep Learning Architecture (DLA) is constantly evolving, and lots of new techniques have been invented to improve the efficiency of DLA. The performance of PipeCNN is no longer comparable to the state-of-the-art designs. Therefore, the current goal of this project is to provide a complete design that can be used to learn DLA and try out new ideas. 

This following table lists the performance and cost information on some of the boards we used as a reference. For each FPGA device, one needs to perform design space exploration (with hardware parameters VEC_SIZE, LANE_NUM and CONV_GP_SIZE_X) to find the optimal design that maximizes the throughput or minimizes the excution time. Suggested hardware parameters for the above boards are summarized [here](https://github.com/doonny/PipeCNN/tree/master/documents). Since we are constantly optimzing the design and updating the codes, the performance data in the following table might be out-dated, and please use the latest version to get the exect data. We welcome other vendors/researches to provide the latest performance and cost information on other FPGA platforms/boards.

| Boards     | Excution Time* | Batch Size | DSP Consumed |  Frequency|
| :--------: |--------------:| ----------:| ------------:|----------:|
| DE5a-net-ddr4    |          -- |         -- |           --|     --|

*Note: ResNet-50 was used as the benchmark. Image size is 227x227x3.

## Citation
Please kindly cite our work of PipeCNN if it helped your research:
```
Dong Wang, Ke Xu and Diankun Jiang, “PipeCNN: An OpenCL-Based Open-Source FPGA Accelerator for Convolution Neural Networks”, FPT 2017.
```
Architectural and algorithm level optimizations can be conducted to further improve the performance of PipeCNN. We list a few latest research achievements that are based on PipeCNN for reference:
* Improving the throughput by introducing a new opencl-friendly sparse-convolution algorithm
```
Dong Wang, Ke Xu, Qun Jia and  Soheil Ghiasi, “ABM-SpConv: A Novel Approach to FPGA-Based Acceleration of Convolutional Neural Network Inference”, DAC 2019.
```

## Contributors

The following people have also contributed to this project:

[Diankun Jiang](https://github.com/dkjiang2018), [Ke Xu](https://github.com/xuke225), Qun Jia, Jianjing An, Xiaoyun Wang, Shihang Fu, Zhihong Bai, Dezheng Zhang.

## Research Opportunity
Our Research Lab is also looking for outstanding students who are interested in designing hardware accelerator for deep-learning algorithms on FPGAs. Please send me a e-mail if you are interested.

## Related Works
There are other FPGA accelerators that also adopt HLS-based design scheme. Some brilliant works are listed as follow. Note that PipeCNN is the first, and only one that is Open-Source （￣︶￣）↗
* U. Aydonat, S. O'Connell, D. Capalija, A. C. Ling, and G. R. Chiu. "An OpenCL™ Deep Learning Accelerator on Arria 10," *in Proc. FPGA 2017*.
* N. Suda, V. Chandra, G. Dasika, A. Mohanty, Y. F. Ma, S. Vrudhula, J. S. Seo, and Y. Cao, "Throughput-Optimized OpenCL-based FPGA accelerator for large-scale convolutional neural networks," *in Proc. FPGA 2016*.
* C. Zhang, P. Li, G. Sun, Y. Guan, B. J. Xiao, and J. Cong, "Optimizing FPGA-based accelerator design for deep convolutional neural networks," *in Proc. FPGA 2015*.
