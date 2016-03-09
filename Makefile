all: em2.cu
	nvcc -o em em2.cu
clean:
	rm em
