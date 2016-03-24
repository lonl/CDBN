function nvmex(cuFileName)
%NVMEX Compiles and links a CUDA file for MATLAB usage
% NVMEX(FILENAME) will create a MEX-File (also with the name FILENAME) by
% invoking the CUDA compiler, nvcc, and then linking with the MEX
% function in MATLAB.


if ispc % Windows
    %CUDA_DIR     = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0"
    %CUDA_NVCC    = $(CUDA_DIR)/bin/nvcc
    %CUDA_CFLAGS  = -I$(CUDA_DIR)/include 
    %CUDA_LDFLAGS = -L$(CUDA_DIR)/lib -L$(CUDA_DIR)/lib64 -lcudart
    
	Host_Compiler_Location = '-ccbin "D:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64"';
	CUDA_INC_Location = ['"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include"'];
    CUDA_SAMPLES_Location =['"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\common\inc"'];
    PIC_Option = '';
    if ( strcmp(computer('arch'),'win32') ==1)
        machine_str = ' --machine 32 ';
        CUDA_LIB_Location = ['"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\lib\Win32"'];
    elseif  ( strcmp(computer('arch'),'win64') ==1)
        machine_str = ' --machine 64 -arch=sm_30 ';
        CUDA_LIB_Location = ['"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\lib\x64"'];
    end
else % Mac and Linux (assuming gcc is on the path)
	
    CUDA_INC_Location = '/usr/local/cuda/include';
    CUDA_SAMPLES_Location = '/usr/local/cuda/samples/common/inc';
	Host_Compiler_Location = ' ';
	PIC_Option = ' --compiler-options -fPIC ';
    machine_str = [];
    if ( strcmp(computer('arch'),'win32') ==1)
        CUDA_LIB_Location = '/usr/local/cuda/lib';
    elseif  ( strcmp(computer('arch'),'win64') ==1)
        CUDA_LIB_Location = '/usr/local/cuda/lib64';
    end
end
% !!! End of things to modify !!!
[~, filename] = fileparts(cuFileName);
nvccCommandLine = [ ...
'nvcc --compile ' Host_Compiler_Location ' ' ...
'-o '  filename '.o ' ...
machine_str PIC_Option ...
' -I' '"' matlabroot '/extern/include "' ...
' -I' CUDA_INC_Location  ...
' "' cuFileName '" ' 
 ];
mexCommandLine = ['mex ' filename '.o'  ' -L' CUDA_LIB_Location  ' -lcudart'];
disp(nvccCommandLine);
warning off;
status = system(nvccCommandLine);
status
warning on;
if status < 0
	error 'Error invoking nvcc';
end
disp(mexCommandLine);
eval(mexCommandLine);
end
