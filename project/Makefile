#set source and target for host
HOST_EXE  = run.exe
HOST_SRCS = ./host/main.cpp ../common/ocl_util.cpp ../common/timer.cpp
HOST_OBJS = $(HOST_SRCS:%.cpp=%.o)
HOST_INCS = ../common
HOST_DEBUG =

#choose the BSP you would like to use
#BSP_DIR = /home/fpga/intelFPGA_pro/19.1/hld/board/a10_ref
#BSP_DIR = /home/fpga/intelFPGA_pro/19.3/hld/board/a10_ref
#BSP_DIR = /home/fpga/intelFPGA_pro/BSP/de5a_net_ddr4
BSP_DIR = C:\intelFPGA_pro\19.3\hld\board\a10_ref
#BSP_DIR = /home/fpga/intelFPGA_pro/BSP/f10a_bsp_181/bsp_1801/f10a_remote_bsp

#if you are loading images from OpenCV interfaces please set to 1
USE_OPENCV = 0

#set source and target for device
#supported vendors are xilinx, altera, amd
VENDOR = altera
#select the host archecture  x86|arm32
PLATFORM = x86
#optional flows are hw|hw_emu|sw_emu|report|profile|rtl|hw_sim
#hw      -> generate hw for deployment
#hw_emu  -> generate waveform for SDAccel
#sw_emu  -> perform software emulation
#hw_sim  -> co-sim with Modelsim for Intel OpenCL SDK
#report  -> generate aoco and report(only for std version) for fast evaluation
#rtl     -> generate aocr and report(only for pro version) for fast evaluation
#profile -> insert performance counters in fpga for dynamic profiling
#init    -> initialization for incremental hw compilation with reduced time
#fast    -> fast incremental compilation with reduced time
FLOW        = rtl
#select which kernel file is used
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
ifeq ($(FLOW),report)
DEV_EXE = $(KERNEL_NAME).aoco
else ifeq ($(FLOW),rtl)
DEV_EXE = $(KERNEL_NAME).aocr
else
DEV_EXE = $(KERNEL_NAME).aocx
endif
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
#for opencv_2.x
#OCV_LIBS =  -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml
#for opencv_3.x
OCV_LIBS =  -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_ml
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

#select compile for sw_emu or hw (used verion >= 19.3)
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
#memory optimization for xilinx
ifeq ($(MAX_MEM_PORT), 1)
OCCFLAGS += --max_memory_ports all
OCCFLAGS += --memory_port_data_width all:$(MEM_PORT_SIZE)
endif
#altera
else ifeq ($(VENDOR),altera)
OCC = aoc
ifeq ($(FLOW),sw_emu)
OCCFLAGS = -v -report -march=emulator -emulator-channel-depth-model=strict -I device/RTL -L device/RTL -l rtl_lib.aoclib -board-package=$(BSP_DIR)
else ifeq ($(FLOW),hw)
OCCFLAGS = -v -report -seed=3  -I device/RTL -L device/RTL -l rtl_lib.aoclib -board-package=$(BSP_DIR)
else ifeq ($(FLOW),init)
OCCFLAGS = -v -report -seed=3  -incremental -I device/RTL -L device/RTL -l rtl_lib.aoclib -board-package=$(BSP_DIR)
else ifeq ($(FLOW),fast)
OCCFLAGS = -v -report -seed=3  -incremental -fast-compile -I device/RTL -L device/RTL -l rtl_lib.aoclib -board-package=$(BSP_DIR)
else ifeq ($(FLOW),profile)
OCCFLAGS = -v -report -profile -I device/RTL -L device/RTL -l rtl_lib.aoclib -board-package=$(BSP_DIR)
else ifeq ($(FLOW),report)
OCCFLAGS = -v -c -report -I device/RTL -L device/RTL -l rtl_lib.aoclib -board-package=$(BSP_DIR)
else ifeq ($(FLOW),rtl)
OCCFLAGS = -v -rtl -report -fmax=300  -I device/RTL -L device/RTL -l rtl_lib.aoclib -board-package=$(BSP_DIR)
else ifeq ($(FLOW),hw_sim)
OCCFLAGS = -v -report -march=simulator -ghdl -I device/RTL -L device/RTL -l rtl_lib.aoclib -board-package=$(BSP_DIR)
endif
# misc options for altera
# add support for arbitrary precision integers
OCCFLAGS += -I $(INTELFPGAOCLSDKROOT)/include/kernel_headers
#amd
else ifeq ($(VENDOR),amd)
OCC =
OCCFLAGS =
endif

#debug option
ifeq ($(KERNEL_DEBUG),1)
	OCCFLAGS += -g
else
	OCCFLAGS += -g0
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

.PHONY: run
run:
	./run.exe $(KERNEL_NAME).aocx

.PHONY: log
log:
	./run.exe $(KERNEL_NAME).aocx | tee >run.log

.PHONY: clean
clean:
	rm -rf *.sof *.xclbin *.rpt *.sv *.efisim *.tmp *.temp *.txt *.json *.dir *.aocx *.aoco *.aocr *.source *.exe xocc* sdaccel* ./device/*.ll ./host/*.o ../common/*.o *.jou *.mon *.cl
