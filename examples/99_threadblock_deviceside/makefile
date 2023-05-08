threadblock-device: threadblock-device.cu
	nvcc $^ -I../../include -I../../tools/util/include/ -arch sm_86 -o $@

clean:
	rm -rf threadblock-device *.o