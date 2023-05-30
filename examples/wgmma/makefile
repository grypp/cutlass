SRC=cutlass-runner.cu
INC=-I../../include -I../../tools/util/include -I../common 
FLAGS= --expt-relaxed-constexpr  $(INC)

all: kernelMultisage.ptx tma.ptx warpSpecialized.ptx warpSpecialized_epitma.ptx pingpong.ptx pingpong_epitma.ptx

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

clean:
	rm -f *.ptx