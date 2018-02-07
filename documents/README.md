# User Instructions
---

## How to run PipeCNN
Before starting to use this project, you need to install Altera or Xilinx's OpenCL SDK toolset on a Linux desktop computer, on which a supported FPGA board is also correctly installed. Clone PipeCNN from [github](https://github.com/doonny/PipeCNN), and download the test vector and golden reference files from PipeCNN's own [ModelZoo](https://github.com/doonny/PipeCNN/tree/master/data). Put all the data files in the ./data folder.

**For Altera users**, first enter the ./RTL folder, run the makefile (simply type *make*). This would generate the necessary RTL libraries used by PipeCNN. Secondly, back to the main project folder, run the main makefile provided, and it will take around one hour to finish all the compilations. Finally, there will be two files generated as follow:
* *run.exe* (host executable)
* *conv.aocx* (fpga bitstream)

Simply start the accelerator by typing
```
./run.exe conv.aocx
```

The results will be like this:
```
***************************************************
PipeCNN: An OpenCL-Based FPGA Accelerator for CNNs 
***************************************************

61063552 total weights read 
154587 bytes image read 
1024 total output reference read 


Platform: Altera SDK for OpenCL
Using 1 device(s)
  Device 0: de1soc_sharedonly : Cyclone V SoC Development Kit
Device OpenCL Version: OpenCL 1.0 Altera SDK for OpenCL, Version 16.0
Device Max Compute Units: 1
Device Max WorkGroup Size: 2147483647
Device Max WorkItem Size: 2147483647
Device Global Memory Size: 512 MBytes
Device Local Memory Size: 16 KBytes
Device Max Clock Freq: 1000 Mhz

Loading kernel/binary from file conv.aocx

Executing Layer 1:

Launching single work-item kernel winbuffer

Launching single work-item kernel Conv

Launching single work-item kernel Pooling

Launching kernel MemWr with local size: 1, 1, 8  (global size: 27, 27, 96)

Launching kernel lrn with local size: 1, 1, 12  (global size: 27, 27, 12)

Executing Layer 2:

Launching single work-item kernel winbuffer

Launching single work-item kernel Conv

Launching single work-item kernel Pooling

Launching kernel MemWr with local size: 1, 1, 8  (global size: 13, 13, 256)

Launching kernel lrn with local size: 1, 1, 32  (global size: 13, 13, 32)

Executing Layer 3:

Launching single work-item kernel winbuffer

Launching single work-item kernel Conv

Launching kernel MemWr with local size: 1, 1, 8  (global size: 13, 13, 384)

Executing Layer 4:

Launching single work-item kernel winbuffer

Launching single work-item kernel Conv

Launching kernel MemWr with local size: 1, 1, 8  (global size: 13, 13, 384)

Executing Layer 5:

Launching single work-item kernel winbuffer

Launching single work-item kernel Conv

Launching single work-item kernel Pooling

Launching kernel MemWr with local size: 1, 1, 8  (global size: 6, 6, 256)

Executing Layer 6:

Launching single work-item kernel winbuffer

Launching single work-item kernel Conv

Launching kernel MemWr with local size: 1, 1, 8  (global size: 1, 1, 4096)

Executing Layer 7:

Launching single work-item kernel winbuffer

Launching single work-item kernel Conv

Launching kernel MemWr with local size: 1, 1, 8  (global size: 1, 1, 4096)

Executing Layer 8:

Launching single work-item kernel winbuffer

Launching single work-item kernel Conv

Launching kernel MemWr with local size: 1, 1, 8  (global size: 1, 1, 1024)

Copyed all batched results from fc_2 buffers.

Done !!!


-------------------

Performance Summary

Total runtime: 0.154791s 

Kernel runtime summary:
  Layer-1:
    Prepare: 0.043391s

    MemRd: 41.686 ms
    Conv : 41.557 ms
    Pool : 41.491 ms
    MemWr: 41.418 ms
    Lrn  : 1.197 ms
  Layer-2:
    Prepare: 0.034802s

    MemRd: 34.120 ms
    Conv : 33.993 ms
    Pool : 33.919 ms
    MemWr: 33.848 ms
    Lrn  : 0.416 ms
  Layer-3:
    Prepare: 0.023367s

    MemRd: 23.173 ms
    Conv : 23.057 ms
    Pool : 0.000 ms
    MemWr: 22.985 ms
    Lrn  : 0.000 ms
  Layer-4:
    Prepare: 0.017615s

    MemRd: 17.423 ms
    Conv : 17.307 ms
    Pool : 0.000 ms
    MemWr: 17.232 ms
    Lrn  : 0.000 ms
  Layer-5:
    Prepare: 0.011972s

    MemRd: 11.769 ms
    Conv : 11.631 ms
    Pool : 11.540 ms
    MemWr: 11.461 ms
    Lrn  : 0.000 ms
  Layer-6:
    Prepare: 0.014695s

    MemRd: 14.493 ms
    Conv : 14.364 ms
    Pool : 0.000 ms
    MemWr: 14.279 ms
    Lrn  : 0.000 ms
  Layer-7:
    Prepare: 0.006769s

    MemRd: 6.565 ms
    Conv : 6.433 ms
    Pool : 0.000 ms
    MemWr: 6.353 ms
    Lrn  : 0.000 ms
  Layer-8:
    Prepare: 0.001983s

    MemRd: 1.782 ms
    Conv : 1.648 ms
    Pool : 0.000 ms
    MemWr: 1.558 ms
    Lrn  : 0.000 ms

Total kernel runtime 149.988 ms 
Batch size = 1, average process time per batch: 149.988 ms 

Start verifying results ...
Selected item = 0 from the combined batch results in fc buffers

Check Pass !!!

The inference result is n02123045 tabby, tabby cat   (the prob is 56.00) 

```

**For Xilinx users**, you need to use Pipes to replace Channels. However, SDx does not support vectorized data type for Pipes, so you need to generate paralelled Pipe instances in a separate file "pipe.cl". To generate this file, use the script "pipe_gen.py" provided. Simply run the following command:
```
python pipe_gen.py [lane_num] [vec_size]
```
After generating the correct pipe.cl file, directly run the makefile should generate everything. Before running the program, remember to set the correct enviroment by using the scripts provided.
* setup_sdx_hw.sh (run PipeCNN on FPGAs)
* setup_sdx_sw_emu.sh (run software emulation)

However, it is always recommanded to use the IDE enviroment rather than makefile-based flow.

### Notes

* Current host code only read one image file (in binary or .jpg) which is reused for each batch process.
* If you are using ARM-based SoC FPGA devices, please change *PLATFORM = x86* in the makefile to *arm32*.
* If you want to run software simulations, please change *FLOW = hw* in the makefile to *sw_emu*, and source setup_emu.sh before running.
---

## Configurations
**HW Configuration.** Configuration of a new FPGA accelerator with different performance and hardware resource utilizations is controlled by a header file located in *device/hw_param.cl*. Change the following macros
* VEC_SIZE
* LANE_NUM
* CONV_GP_SIZE_X

to appropriate ones. The default setting is VEC_SIZE=8, LANE_NUM=16, CONV_GP_SIZE_X=7 which achieves the shortest classification time on the DE5-net board. To obtain the optimal results (best performance or smallest cost), you need to perform design space explorations by implementing PipeCNN with different configurations of the three parameters, and find the one as you needed. Please refer to our acdamic papers for more detailed information.

**SW Configuration.** Configuration of different CNN models is done by a header file located in *host/layer_config.h*. Select one of the model configurations provided and recompile the host before running the test. Currently, the following models have been tested:
* AlexNet (CaffeNet)
* Vgg-16

