export LD_LIBRARY_PATH=$XILINX_OPENCL/runtime/lib/x86_64:$LD_LIBRARY_PATH
export PATH=$XILINX_OPENCL/runtime/bin:$PATH
export XCL_EMULATION_MODE=sw_emu
emconfigutil --platform xilinx:kcu1500:4ddr-xpr:4.0
unset XILINX_SDACCEL
unset XILINX_OPENCL
