SRC=cutlass_runner_f16n_f16n_f16n.cu
INC=-I../../include -I../../tools/util/include -I../common 
LIBS = -L"/usr/local/cuda-12.1/targets/x86_64-linux/lib/stubs" -L"/usr/local/cuda-12.1/targets/x86_64-linux/lib" -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl
XFLAGS=-Xcompiler=-fPIE -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -std=c++17   
FLAGS= $(XFLAGS) --expt-relaxed-constexpr  $(INC) $(LIBS)

ptx: kernelMultisage.ptx tma.ptx warpSpecialized.ptx warpSpecialized_epitma.ptx pingpong.ptx pingpong_epitma.ptx wgmmaf16.ptx wgmmaf32.ptx
exe : kernelMultisage.exe tma.exe warpSpecialized.exe warpSpecialized_epitma.exe pingpong.exe pingpong_epitma.exe 
exe_debug : kernelMultisage_debug.exe tma_debug.exe warpSpecialized_debug.exe warpSpecialized_epitma_debug.exe pingpong_debug.exe pingpong_epitma_debug.exe

wgmmaf16.exe: wgmma_m64n128k16_f16n_f16n_f16n.cu
	nvcc -arch=sm_90a wgmma_m64n128k16_f16n_f16n_f16n.cu -o $@ -DDEBUG3

wgmmaf32.exe: wgmma_m64n128k16_f16n_f16n_f16n.cu
	nvcc -DF32 -arch=sm_90a wgmma_m64n128k16_f16n_f16n_f16n.cu -o $@ -DDEBUG3

wgmmaf16.ptx: wgmma_m64n128k16_f16n_f16n_f16n.cu
	nvcc -ptx -arch=sm_90a wgmma_m64n128k16_f16n_f16n_f16n.cu -o $@ 

wgmmaf32.ptx: wgmma_m64n128k16_f16n_f16n_f16n.cu
	nvcc -ptx -DF32 -arch=sm_90a wgmma_m64n128k16_f16n_f16n_f16n.cu -o $@ 

kernelMultisage.ptx : $(SRC)
	nvcc -arch=sm_90a -ptx $(SRC) -o $@ $(FLAGS) -DKERNEL_MULTISTAGE

tma.ptx: $(SRC)
	nvcc -arch=sm_90a -ptx $(SRC) -o $@ $(FLAGS) -DKERNEL_TMA 

warpSpecialized.ptx: $(SRC)
	nvcc -arch=sm_90a -ptx $(SRC) -o $@ $(FLAGS) -DKERNEL_WARPSPECIALIZED

warpSpecialized_epitma.ptx: $(SRC)
	nvcc -arch=sm_90a -ptx $(SRC) -o $@ $(FLAGS) -DKERNEL_WARPSPECIALIZED_EPITMA

pingpong.ptx: $(SRC)
	nvcc -arch=sm_90a -ptx $(SRC) -o $@ $(FLAGS) -DKERNEL_PINGPONG

pingpong_epitma.ptx: $(SRC)
	nvcc -arch=sm_90a -ptx $(SRC) -o $@ $(FLAGS) -DKERNEL_PINGPONG_EPITMA

# ////

kernelMultisage_debug.exe : $(SRC)
	nvcc -arch=sm_90a  $(SRC) -o $@ $(FLAGS) -DKERNEL_MULTISTAGE -DDEBUG -DDEBUG_DESCRIPTOR -DCUTLASS_DEBUG_TRACE_LEVEL=10 -lineinfo

tma_debug.exe: $(SRC)
	nvcc -arch=sm_90a  $(SRC) -o $@ $(FLAGS) -DKERNEL_TMA  -DDEBUG  -DDEBUG_DESCRIPTOR -DCUTLASS_DEBUG_TRACE_LEVEL=10

warpSpecialized_debug.exe: $(SRC)
	nvcc -arch=sm_90a  $(SRC) -o $@ $(FLAGS) -DKERNEL_WARPSPECIALIZED -DDEBUG  -DDEBUG_DESCRIPTOR -DCUTLASS_DEBUG_TRACE_LEVEL=10

warpSpecialized_epitma_debug.exe: $(SRC)
	nvcc -arch=sm_90a  $(SRC) -o $@ $(FLAGS) -DKERNEL_WARPSPECIALIZED_EPITMA -DDEBUG  -DDEBUG_DESCRIPTOR -DCUTLASS_DEBUG_TRACE_LEVEL=10

pingpong_debug.exe: $(SRC)
	nvcc -arch=sm_90a  $(SRC) -o $@ $(FLAGS) -DKERNEL_PINGPONG -DDEBUG  -DDEBUG_DESCRIPTOR -DCUTLASS_DEBUG_TRACE_LEVEL=10

pingpong_epitma_debug.exe: $(SRC)
	nvcc -arch=sm_90a  $(SRC) -o $@ $(FLAGS) -DKERNEL_PINGPONG_EPITMA -DDEBUG  -DDEBUG_DESCRIPTOR -DCUTLASS_DEBUG_TRACE_LEVEL=10

# ////

kernelMultisage.exe : $(SRC)
	nvcc -arch=sm_90a  $(SRC) -o $@ $(FLAGS) -DKERNEL_MULTISTAGE

tma.exe: $(SRC)
	nvcc -arch=sm_90a  $(SRC) -o $@ $(FLAGS) -DKERNEL_TMA  

warpSpecialized.exe: $(SRC)
	nvcc -arch=sm_90a  $(SRC) -o $@ $(FLAGS) -DKERNEL_WARPSPECIALIZED 

warpSpecialized_epitma.exe: $(SRC)
	nvcc -arch=sm_90a  $(SRC) -o $@ $(FLAGS) -DKERNEL_WARPSPECIALIZED_EPITMA 

pingpong.exe: $(SRC)
	nvcc -arch=sm_90a  $(SRC) -o $@ $(FLAGS) -DKERNEL_PINGPONG 

pingpong_epitma.exe: $(SRC)
	nvcc -arch=sm_90a  $(SRC) -o $@ $(FLAGS) -DKERNEL_PINGPONG_EPITMA 


clean:
	rm -f *.ptx *.exe