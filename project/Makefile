#set source and target for host
HOST_EXE  = run.exe
HOST_SRCS = ./host/main.cpp ../common/ocl_util.cpp ../common/timer.cpp
HOST_OBJS = $(HOST_SRCS:%.cpp=%.o)
HOST_INCS = ../common
HOST_DEBUG =

#if you are loading images from OpenCV interfaces please set to 1
USE_OPENCV = 0

#set source and target for device
#supported vendors are xilinx, altera, amd
VENDOR = altera
#select the host archecture  x86|arm32
PLATFORM = x86
#optinal flows are hw|hw_emu|sw_emu
FLOW        = sw_emu
ifeq ($(VENDOR),xilinx)
KERNEL_SRCS = ./device/conv_pipe_xilinx.cl
else
KERNEL_SRCS = ./device/conv_pipe.cl
endif
KERNEL_NAME = conv
KERNEL_DEFS =
KERNEL_INCS = 
KERNEL_DEBUG = 1
#additional pragma for xilinx
CU_NUM = 1
MAX_MEM_PORT = 0
MEM_PORT_SIZE = 512

#select the device binary name
ifeq ($(VENDOR),xilinx)
DEV_EXE = $(KERNEL_NAME).xclbin
else ifeq ($(VENDOR),altera)
DEV_EXE = $(KERNEL_NAME).aocx
else ifeq ($(VENDOR),amd)
DEV_EXE =
endif

#host compiler options
CROSS-COMPILE = arm-linux-gnueabihf-
ifeq ($(PLATFORM),x86)
CXX := g++
ifeq ($(USE_OPENCV),1)
#add your OpenCV PATH here
OCV_INCLUDES = -I/usr/local/include/
OCV_LIBDIRS = -L/usr/local/lib 
OCV_LIBS =  -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml
endif
else ifeq ($(PLATFORM),arm32)
CXX := $(CROSS-COMPILE)g++
ifeq ($(USE_OPENCV),1)
#add your cross compile OpenCV PATH here
OCV_INCLUDES = -I/usr/local/opencv-arm/include/
OCV_LIBDIRS = -L/usr/local/opencv-arm/lib 
OCV_LIBS =  -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml
endif
endif

#select whether use OpenCV or not
ifeq ($(USE_OPENCV),1)
CXXFLAGS = -g -Wall -std=c++11 -DUSE_OPENCV
else
CXXFLAGS = -g -Wall -std=c++11
endif

#select whether manually launch free-run kernels
ifeq ($(FLOW),sw_emu)
CXXFLAGS += -DSW_EMU
endif

ifeq ($(VENDOR),xilinx)
COMP_CONFIG = -I$(XILINX_SDX)/runtime/include/1_2 -I$(HOST_INCS)  -DFPGA_DEVICE -DXILINX
LINK_CONFIG = -L$(XILINX_SDX)/runtime/lib/x86_64 -lxilinxopencl
else ifeq ($(VENDOR),altera)
COMP_CONFIG = $(shell aocl compile-config) -I$(HOST_INCS)  -DFPGA_DEVICE
LINK_CONFIG = $(shell aocl link-config)
else ifeq ($(VENDOR),amd)
DEV_EXE =
endif

#opencl compiler options
#xilinx
ifeq ($(VENDOR),xilinx)
OCC = $(XILINX_SDX)/bin/xocc -DXILINX
OCCFLAGS = --platform xilinx:kcu1500:4ddr-xpr:4.0 -t $(FLOW) -O2  --kernel_frequency 200  --save-temps  --report system \
--memory_port_data_width memWrite:32 \
--memory_port_data_width memRead:512 \
--nk memWrite:1  --nk coreConv:1  --nk lrn:1  --nk maxPool:1  --nk memRead:1 \
#--xp misc:map_connect=add.kernel.memRead_1.M_AXI_GMEM0.core.OCL_REGION_0.M00_AXI\
#--xp misc:map_connect=add.kernel.memRead_1.M_AXI_GMEM1.core.OCL_REGION_0.M01_AXI\
#--xp misc:map_connect=add.kernel.memRead_1.M_AXI_GMEM2.core.OCL_REGION_0.M02_AXI\
#--xp misc:map_connect=add.kernel.memWrite_1.M_AXI_GMEM0.core.OCL_REGION_0.M00_AXI\
#--xp misc:map_connect=add.kernel.lrn_1.M_AXI_GMEM.core.OCL_REGION_0.M00_AXI
#--max_memory_ports lrn\
#--xp misc:map_connect=add.kernel.lrn_1.M_AXI_GMEM0.core.OCL_REGION_0.M00_AXI\
#--xp misc:map_connect=add.kernel.lrn_1.M_AXI_GMEM1.core.OCL_REGION_0.M01_AXI
#memory optimization for xilinx
ifeq ($(MAX_MEM_PORT), 1)
OCCFLAGS += --max_memory_ports all
OCCFLAGS += --memory_port_data_width all:$(MEM_PORT_SIZE)
endif
#altera
else ifeq ($(VENDOR),altera)
OCC = aoc
ifeq ($(FLOW),sw_emu)
OCCFLAGS = -v --report -march=emulator -I device/RTL -L device/RTL -l rtl_lib.aoclib 
else ifeq ($(FLOW),hw)
OCCFLAGS = -v --report --profile -I device/RTL -L device/RTL -l rtl_lib.aoclib 
endif
#amd
else ifeq ($(VENDOR),amd)
OCC =
OCCFLAGS =
endif

#debug option
ifeq ($(KERNEL_DEBUG),1)
	OCCFLAGS += -g
endif

.PHONY: all
all: host fpga

.PHONY: host
host: $(HOST_EXE)

.PHONY: fpga
fpga: $(DEV_EXE)

$(HOST_EXE): $(HOST_OBJS)
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(HOST_OBJS) -o $@ $(LINK_CONFIG) $(OCV_LIBS)
	
%.o: %.cpp
	$(CXX) $(OCV_LIBDIRS) $(OCV_INCLUDES) $(CXXFLAGS) -c $< -o $@ $(COMP_CONFIG) $(OCV_LIBS)

$(DEV_EXE): $(KERNEL_SRCS)
	$(OCC) $(OCCFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf *.xclbin *.rpt *.xtxt *.json *.dir *.aocx *.aoco *.exe xocc* sdaccel* ./device/*.ll ./host/*.o ../common/*.o *.log *.jou *.mon
