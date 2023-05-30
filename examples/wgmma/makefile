SRC=cutlass-runner.cu
FLAGS=-I../../include -I../../tools/util/include -I../common 

all: kernelMultisage.ptx tma.ptx warpSpecialized.ptx warpSpecialized_epitma.ptx pingpong.ptx pingpong_epitma.ptx

kernelMultisage.ptx : $(SRC)
	nvcc -arch=sm_90a -ptx $(SRC) -o $@ $(FLAGS) -DMULTISTAGE

tma.ptx: $(SRC)
	nvcc -arch=sm_90a -ptx $(SRC) -o $@ $(FLAGS) -DTMA

warpSpecialized.ptx: $(SRC)
	nvcc -arch=sm_90a -ptx $(SRC) -o $@ $(FLAGS) -DWARPSPECIALIZED

warpSpecialized_epitma.ptx: $(SRC)
	nvcc -arch=sm_90a -ptx $(SRC) -o $@ $(FLAGS) -DWARPSPECIALIZED_EPITMA

pingpong.ptx: $(SRC)
	nvcc -arch=sm_90a -ptx $(SRC) -o $@ $(FLAGS) -DPINGPONG

pingpong_epitma.ptx: $(SRC)
	nvcc -arch=sm_90a -ptx $(SRC) -o $@ $(FLAGS) -DPINGPONG_EPITMA

clean:
	rm -f kernelMultisage.ptx tma.ptx warpSpecialized.ptx warpSpecialized_epitma.ptx pingpong.ptx pingpong_epitma.ptx