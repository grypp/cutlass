SRC=cutlass_runner_f16n_f16n_f16n.cu
INC=-I../../include -I../../tools/util/include -I../common 
FLAGS= --expt-relaxed-constexpr  $(INC)

ptx: kernelMultisage.ptx tma.ptx warpSpecialized.ptx warpSpecialized_epitma.ptx pingpong.ptx pingpong_epitma.ptx
exe : kernelMultisage.exe tma.exe warpSpecialized.exe warpSpecialized_epitma.exe pingpong.exe pingpong_epitma.exe

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

kernelMultisage.exe : $(SRC)
	nvcc -arch=sm_90a  $(SRC) -o $@ $(FLAGS) -DKERNEL_MULTISTAGE -DDEBUG

tma.exe: $(SRC)
	nvcc -arch=sm_90a  $(SRC) -o $@ $(FLAGS) -DKERNEL_TMA  -DDEBUG

warpSpecialized.exe: $(SRC)
	nvcc -arch=sm_90a  $(SRC) -o $@ $(FLAGS) -DKERNEL_WARPSPECIALIZED -DDEBUG

warpSpecialized_epitma.exe: $(SRC)
	nvcc -arch=sm_90a  $(SRC) -o $@ $(FLAGS) -DKERNEL_WARPSPECIALIZED_EPITMA -DDEBUG

pingpong.exe: $(SRC)
	nvcc -arch=sm_90a  $(SRC) -o $@ $(FLAGS) -DKERNEL_PINGPONG -DDEBUG

pingpong_epitma.exe: $(SRC)
	nvcc -arch=sm_90a  $(SRC) -o $@ $(FLAGS) -DKERNEL_PINGPONG_EPITMA -DDEBUG

clean:
	rm -f *.ptx *.exe