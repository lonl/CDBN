# Convolutional Deep Belief Networks with 'MATLAB','MEX','CUDA' versions

This program is an implementation of Convolutional Deep Belief Networks. In this code, the binary and Gaussian visable types are both supported. In addition, CUDA acceleration is also included. We provide some demo programs to show the usage of the code. 



## Requirement
* OS: Ubuntu 10.04 (64-bit) (only test on this plateform)  
* GNU C/C++ compiler
* Matlab
* CUDA 5.0 or above


## Build
* Change the path to 'CDBN/toolbox/CDBNLIB/mex'
* edit 'Makefile', modify 'MATLAB_DIR' and 'CUDA_DIR' to your correct path.
* `make`


## Run the program
* run 'setup_toolbox.m';
* run 'DemoCDBN_Binary_2D.m' or 'DemoCDBN_Gaussian_2D.m'
 
           
## Experiments
We have conducted classification experiments with 'Convolutional Deep Belief Networks', 'Deep Belief Networks', and 'Directed Softmax' in mnist data (2000 train data & 2000 test data). The detail parameters of these three ways can be found in code. 

The comparison results (accuracy) are as follows:

 No noise added in test data:
     CDBN:     95.1%
     DBN:      91.5%
     Softmax:  87.7%

 10% noise added in test data:
     CDBN:     92.8%
     DBN:      86.7%
     Softmax:  83.2%

 20% noise added in test data:
     CDBN:     84.4%
     DBN:      60.1%
     Softmax:  74.7%

## Note
* Different computation methods can be selected. Currently, matlab matrix computation, MEX, CDUA are supported. You can change the computation method globaly in 'CDBN/toolbox/CDBNLIB/default_layer2D.m' by select one of the methods.
`
 layer.matlab_use    = 0;
 layer.mex_use       = 1;
 layer.cuda_use      = 0;
`

or you can change the computation method in the layer defination, for example, you can add above lines to 'DemoCDBN_Binary_2D.m' at layer 1's defination as:
`
 layer{1}.matlab_use    = 0;
 layer{1}.mex_use       = 0;
 layer{1}.cuda_use      = 1;
`

* The acceleration effect of 'CUDA' version is not obvious in first layer. But it may be better in the later layer for big size pictures. 

           
## Connection
If you have any problem, or you have some suggestions for this code, please contact me: hanpc839874404@163.com, thank you very much!

