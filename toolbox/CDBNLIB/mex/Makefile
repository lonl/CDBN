###############################################################################
###############################################################################
CC  = gcc
CXX = g++

###############################################################################
###############################################################################
DEFAULT_CFLAGS  = -D_GNU_SOURCE -O3 -DNDEBUG
DEFAULT_LDFLAGS = -lstdc++

###############################################################################
###############################################################################
MATLAB_DIR    = /opt/matlab_r2013a
MATLAB_CFLAGS = -I$(MATLAB_DIR)/extern/include -I$(MATLAB_DIR)/simulink/include \
		-DMATLAB_MEX_FILE -ansi -fexceptions -fPIC -fno-omit-frame-pointer \
		-pthread  -DMX_COMPAT_32 
MATLAB_CUFLAGS = -I$(MATLAB_DIR)/extern/include -I$(MATLAB_DIR)/simulink/include \
		-DMATLAB_MEX_FILE -DMX_COMPAT_32 
MATLAB_LDFLAGS = -pthread -shared -Wl,--version-script,$(MATLAB_DIR)/extern/lib/glnxa64/mexFunction.map \
		-Wl,--no-undefined \
		-Wl,-rpath-link,$(MATLAB_DIR)/bin/glnxa64 \
		-L$(MATLAB_DIR)/bin/glnxa64 -lmx -lmex -lmat -lm 

###############################################################################
###############################################################################
CUDA_DIR     = /opt/cuda-5.0
CUDA_NVCC    = $(CUDA_DIR)/bin/nvcc
CUDA_CFLAGS  = -I$(CUDA_DIR)/include 
CUDA_LDFLAGS = -L$(CUDA_DIR)/lib -L$(CUDA_DIR)/lib64 -lcudart


###############################################################################
###############################################################################
CFLAGS  = $(DEFAULT_CFLAGS) $(MATLAB_CFLAGS) $(CUDA_CFLAGS)
LDFLAGS = $(DEFAULT_LDFLAGS) $(MATLAB_LDFLAGS) $(CUDA_LDFLAGS)
CUFLAGS = -arch=sm_30 --compiler-options -fPIC,$(DEFAULT_CFLAGS) $(MATLAB_CUFLAGS) $(CUDA_CFLAGS)

###############################################################################
###############################################################################
target : crbm2D_batch_mex.mexa64 crbm_forward2D_batch_mex.mexa64 crbm2D_mex_cuda.mexa64 

crbm2D_batch_mex.mexa64 : crbm2D_batch_mex.c
	$(CC) -c $? $(CFLAGS)
	$(CC) crbm2D_batch_mex.o -o crbm2D_batch_mex.mexa64 $(LDFLAGS)

crbm_forward2D_batch_mex.mexa64 : crbm_forward2D_batch_mex.c
	$(CC) -c $? $(CFLAGS)
	$(CC) crbm_forward2D_batch_mex.o -o crbm_forward2D_batch_mex.mexa64 $(LDFLAGS)

crbm2D_mex_cuda.mexa64 : crbm2D_mex_cuda.cu
	$(CUDA_NVCC) -c $? $(CUFLAGS)
	$(CC) crbm2D_mex_cuda.o -o crbm2D_mex_cuda.mexa64 $(LDFLAGS)

clean:
	rm -f *.mex*
