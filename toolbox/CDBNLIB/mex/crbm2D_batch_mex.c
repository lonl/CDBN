#include <math.h>
#include <mex.h>
#include <matrix.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	const mxArray *model, *layer, *batch_data;
	      mxArray *model_new;

    /**@brief
      *Load Data from Matlab
      */

    /*Check the structure of the fisrt input data*/
	if(mxIsStruct(prhs[0]) & mxIsStruct(prhs[1])){
		model       = prhs[0];
		layer       = prhs[1];
		batch_data  = prhs[2];
	}
	else
		mexErrMsgTxt("You must specify two structure arrays!");

	/**@brief
	  *Calculate Gradient
	  *Calculate dW, dV_bias, dH_bias
	  */

	/*** Positive Updata: Bottom-up process ***/

	/* Crbm Inference: Initialize Hidden Sample */
	mxArray        *h_input_array, *h_sample_array,*h_state_array, *h_sam_in_array;
	mwSize         *dim_h;
	double         *h_input, *h_sample,*h_sample_init, *h_state;
    int            ndim_h;

	dim_h          = mxGetDimensions(mxGetField(model,0,"h_input"));
    ndim_h         = mxGetNumberOfDimensions(mxGetField(model,0,"h_input"));

    if (ndim_h == 2){
        dim_h[2] = 1;
        dim_h[3] = 1;
    }
    else if (ndim_h == 3){
        dim_h[3] = 1;
    }

	h_input_array  = mxCreateNumericArray(4,dim_h,mxDOUBLE_CLASS, mxREAL);
	h_input        = mxGetPr(h_input_array);
	h_sample_array = mxCreateNumericArray(4,dim_h,mxDOUBLE_CLASS, mxREAL);
	h_sample       = mxGetPr(h_sample_array);
	h_state_array  = mxCreateNumericArray(4,dim_h,mxDOUBLE_CLASS, mxREAL);
	h_state        = mxGetPr(h_state_array);
	h_sam_in_array = mxCreateNumericArray(4,dim_h,mxDOUBLE_CLASS, mxREAL);
    h_sample_init  = mxGetPr(h_sam_in_array);

	/* Crbm Inference Function */
	crbm_inference2D(h_sample_init,h_input,h_state,model,layer,batch_data);
      
    
	/*** Negative Update: Top-down process ***/

    /* Crbm reconstruction: Initialize Visual Sample*/
    mxArray        *v_input_array, *v_sample_array;
    mwSize         *dim_v1,*dim_v;
    int            ndim_v1;
    double         *v_input, *v_sample;


    dim_v1         = mxGetDimensions(mxGetField(model,0,"v_input"));
    ndim_v1        = mxGetNumberOfDimensions(mxGetField(model,0,"v_input"));

    /*You must pay attention here: if the n_map_v is 1, some wrong will be coming, luckily, I solve it*/

    dim_v          = mxMalloc(sizeof(mwSize)*4);
    dim_v[0]       = dim_v1[0];
    dim_v[1]       = dim_v1[1];

    if (ndim_v1 == 2){
    	dim_v[2] = 1;
        dim_v[3] = 1;
    }
    else if (ndim_v1 == 3){
    	dim_v[2] = dim_v1[2];
        dim_v[3] = 1;
    }
    else{
        dim_v[2] = dim_v1[2];
        dim_v[3] = dim_v1[3];
    }

    v_input_array  = mxCreateNumericArray(4,dim_v,mxDOUBLE_CLASS, mxREAL);  
    v_input        = mxGetPr(v_input_array);
    v_sample_array = mxCreateNumericArray(4,dim_v,mxDOUBLE_CLASS, mxREAL);
    v_sample       = mxGetPr(v_sample_array);

    
	/* Crbm reconstruction: Reconstruct the Visual sample, also called CD algorithm, just once here*/

	crbm_reconstruct2D(v_sample,v_input,h_state,model,layer);

	crbm_inference2D(h_sample,h_input,h_state,model,layer,v_sample_array);


	/* Calculate dW */
	int       ni,N,n_map_v,n_map_h,nh,nv,j,i,jj,ii,id,Hstride,Wstride,Hfilter,Wfilter,Hres,Wres,H,W;
	double    *dW, *dH_bias, *dV_bias,*s_filter,*stride,*data;
	mwSize    *dim_w, *dim_vbias, *dim_hbias;
	mxArray   *dW_array;

	dim_w     = mxGetDimensions(mxGetField(model,0,"W"));
	dW_array  = mxCreateNumericArray(4,dim_w,mxDOUBLE_CLASS, mxREAL);
	dW        = mxGetPr(dW_array);
	dim_vbias = mxGetDimensions(mxGetField(model,0,"dV_bias"));
	dV_bias   = mxGetPr(mxCreateNumericArray(2,dim_vbias,mxDOUBLE_CLASS, mxREAL));
	dim_hbias = mxGetDimensions(mxGetField(model,0,"dH_bias"));
	dH_bias   = mxGetPr(mxCreateNumericArray(2,dim_hbias,mxDOUBLE_CLASS, mxREAL));

	n_map_h   = mxGetScalar(mxGetField(layer,0,"n_map_h"));
	n_map_v   = mxGetScalar(mxGetField(layer,0,"n_map_v"));
	s_filter  = mxGetPr(mxGetField(layer,0,"s_filter"));   
    stride    = mxGetPr(mxGetField(layer,0,"stride"));
    data      = mxGetPr(batch_data);

    Hstride   = stride[0];
    Wstride   = stride[1];
    Hfilter   = s_filter[0];
    Wfilter   = s_filter[1];
    H         = dim_v[0];
    W         = dim_v[1];
    Hres      = dim_h[0];
    Wres      = dim_h[1];
    N         = dim_v[3];


    for(ni = 0; ni < N; ni++){

        for(nh = 0; nh < n_map_h; nh++){
            for (j = 0; j < Wfilter; j++){
                for (i = 0; i < Hfilter; i++){

                    for(nv = 0; nv < n_map_v; nv++){
                        for(jj = 0; jj < Wres; jj++){
                            for (ii = 0; ii < Hres; ii++){

                                id      = i + Hfilter*j + Hfilter*Wfilter*nv + Hfilter*Wfilter*n_map_v*nh;
                                dW[id] += (data[(ii*Hstride+i)+H*(jj*Wstride+j)+H*W*nv+H*W*n_map_v*ni]
                                        * h_sample_init[(ii+Hres*jj)+Hres*Wres*nh+Hres*Wres*n_map_h*ni]
                                        - v_sample[(ii*Hstride+i)+H*(jj*Wstride+j)+H*W*nv+H*W*n_map_v*ni]
                                        * h_sample[(ii+Hres*jj)+Hres*Wres*nh+Hres*Wres*n_map_h*ni]);
                            }
                        }
                    }
                }
            }
        }
    }

	/* With the struct matrix to return: h_sample, v_sample, dW */

	const char *fieldname[] = {"h_sample","h_sample_init","v_sample","dW"};
    mxArray    *struct_array;
    struct_array = plhs[0] = mxCreateStructMatrix(1,1,4,fieldname);
 
    mxSetField(struct_array,0,"h_sample",h_sample_array);
    mxSetField(struct_array,0,"h_sample_init",h_sam_in_array);
    mxSetField(struct_array,0,"v_sample",v_sample_array);
    mxSetField(struct_array,0,"dW",dW_array);
    
    mxFree(dim_v);

	
}


void crbm_inference2D(double h_sample[], double h_input[],double h_state[],
                      const mxArray *model,const mxArray *layer, 
                      const mxArray *batch_data)
{
	int      ni,N,n_map_h, n_map_v, nh,nv,j,i,jj,ii,id,ndim_v1,
             Hstride,Wstride,Hfilter,Wfilter,Hres,Wres,H,W,Hpool,Wpool;
	int      *_id;
    double   *s_filter, *stride, *h, *data, *weights, *h_bias, *block, *pool, *gaussian;
    mwSize   *dim_vi, *dim_hi, *dim_id, *dim_h;
    int      ndim_hi;
    mxChar   *type;

    type     = mxGetChars(mxGetField(layer,0,"type_input"));
	n_map_h  = mxGetScalar(mxGetField(layer,0,"n_map_h"));
	n_map_v  = mxGetScalar(mxGetField(layer,0,"n_map_v"));
    s_filter = mxGetPr(mxGetField(layer,0,"s_filter"));   
    stride   = mxGetPr(mxGetField(layer,0,"stride"));
    data     = mxGetPr(batch_data);
    dim_vi   = mxGetDimensions(batch_data);
    dim_hi   = mxGetDimensions(mxGetField(model,0,"h_input"));
    ndim_hi  = mxGetNumberOfDimensions(mxGetField(model,0,"h_input"));
    gaussian = mxGetPr(mxGetField(model,0,"start_gau"));


    dim_h    = (mwSize*)mxMalloc(sizeof(mwSize)*4);
    dim_h[0] = dim_hi[0];
    dim_h[1] = dim_hi[1];

    if (ndim_hi == 2){
        dim_h[2] = 1;
        dim_h[3] = 1;
    }
    else if (ndim_hi == 3){
        dim_h[2] = dim_hi[2];
        dim_h[3] = 1;
    }
    else{
        dim_h[2] = dim_hi[2];
        dim_h[3] = dim_hi[3];
    }

    weights  = mxGetPr(mxGetField(model,0,"W"));
    h        = mxGetPr(mxCreateNumericArray(4,dim_h,mxDOUBLE_CLASS,mxREAL));
    h_bias   = mxGetPr(mxGetField(model,0,"h_bias"));
    block    = mxGetPr(mxCreateNumericArray(4,dim_h,mxDOUBLE_CLASS,mxREAL));
    pool     = mxGetPr(mxGetField(layer,0,"s_pool"));
    ndim_v1  = mxGetNumberOfDimensions(batch_data);
    mxFree(dim_h);

    if (ndim_v1 == 2 || ndim_v1 == 3)
        N = 1;
    else
        N = dim_vi[3];


    Hstride = stride[0];
    Wstride = stride[1];
    Hfilter = s_filter[0];
    Wfilter = s_filter[1];
    H       = dim_vi[0];
    W       = dim_vi[1];
    Hres    = dim_hi[0];
    Wres    = dim_hi[1];
    Hpool   = pool[0];
    Wpool   = pool[1];

    /*mexPrintf("inference: Hstride = %d, Hfilter = %d H = %d Hres = %d\n",Hstride,Hfilter, H, Hres);*/

    dim_id   = (mwSize*)mxMalloc(sizeof(mwSize)*2);
    dim_id[0]= pool[0]; dim_id[1] = pool[1];
    _id      = mxGetPr(mxCreateNumericArray(2,dim_id,mxUINT32_CLASS,mxREAL));
    mxFree(dim_id);


    for (ni = 0; ni < N; ni++) {  

        for (nh = 0; nh < n_map_h; nh++) {

        	for (j = 0; j < Wres; j++) {
        		for (i = 0; i < Hres; i++) {
        			id = i + Hres * j + Hres * Wres * nh + Hres*Wres*n_map_h*ni;

        			h_input[id] = 0;

                    for (nv = 0; nv < n_map_v; nv++)
        				for (jj = 0; jj < Wfilter; jj++)
        					for (ii = 0; ii < Hfilter; ii++){
        						h_input[id] += data[(i*Hstride+ii)+H*(j*Wstride+jj)+H*W*nv+H*W*n_map_v*ni]
                                       * weights[(ii+Hfilter*jj)+Hfilter*Wfilter*nv+Hfilter*Wfilter*n_map_v*nh];
        					}

        			h_input[id] = h_input[id] + h_bias[nh];

                    /* for crbm blocksum*/

                    if (type[0] == 'B')
                        block[id]   = exp(h_input[id]);
                    if (type[0] == 'G')
                        block[id]   = exp(1.0/(gaussian[0]*gaussian[0])*h_input[id]);

                }
            }


            /* crbm blocksum: hidden activation summation */   	
        	for (j = 0; j < floor(Wres/Wpool); j++) {
        		for (i = 0; i < floor(Hres/Hpool); i++) {

                    double sum = 0.0;
        			for (jj = 0; jj < Wpool; jj++){
        			 	_id[jj*Hpool] = i*Hpool+(j*Wpool+jj)*Hres + Hres*Wres*nh+Hres*Wres*n_map_h*ni;
        			 	sum += block[_id[jj*Hpool]];
        			 	    for (ii = 1; ii < Hpool; ii++){
        			 	    	_id[jj*Hpool+ii] = _id[jj*Hpool+ii-1] + 1;
        			 	    	sum += block[_id[jj*Hpool+ii]];
        			 	    }
        			}

                    bool done  = false;
                    double rnd = rand() % 10000 / 10000.0;
                    double pro_sum = 0.0;
        			for (jj = 0; jj < Hpool*Wpool; jj++){
        				h_sample[_id[jj]] = block[_id[jj]]/(1.0+sum);
        				pro_sum += h_sample[_id[jj]];

        				/* Randomly generate the hidden state: at most one unit is activated */
        				if(done == false){
        					if (pro_sum >= rnd){
        						h_state[_id[jj]] = 1;
        						done = true;
        					}
        				}
        			}

                }        
            }
        }
    }

    return;			
}


void crbm_reconstruct2D(double v_sample[], double v_input[], double h_state[], 
                        const mxArray *model,const mxArray *layer)
{
	int       ni,N,n_map_v,n_map_h,nh,nv,j,i,jj,ii,id,
              Hstride,Wstride,Hfilter,Wfilter,Hres,Wres,H,W,ndim_v1;
	double    *s_filter, *stride, *v, *weights, *v_bias, *h_state_off;
    mwSize    *dim_vi, *dim_hi, *dim_v1, *dim_offset;
    mxChar    *type;

    type      = mxGetChars(mxGetField(layer,0,"type_input"));
    n_map_h   = mxGetScalar(mxGetField(layer,0,"n_map_h"));
	n_map_v   = mxGetScalar(mxGetField(layer,0,"n_map_v"));
    s_filter  = mxGetPr(mxGetField(layer,0,"s_filter"));   
    stride    = mxGetPr(mxGetField(layer,0,"stride"));
    dim_hi    = mxGetDimensions(mxGetField(model,0,"h_input"));
    weights   = mxGetPr(mxGetField(model,0,"W"));
    dim_v1    = mxGetDimensions(mxGetField(model,0,"v_input"));
    ndim_v1   = mxGetNumberOfDimensions(mxGetField(model,0,"v_input"));

    dim_vi    = (mwSize*)mxMalloc(sizeof(mwSize)*4);
    dim_vi[0] = dim_v1[0];
    dim_vi[1] = dim_v1[1];

    if (ndim_v1 == 2){
    	dim_vi[2] = 1;
        dim_vi[3] = 1;
    }
    else if(ndim_v1 == 3){
    	dim_vi[2] = dim_v1[2];
        dim_vi[3] = 1;
    }
    else{
        dim_vi[2] = dim_v1[2];
        dim_vi[3] = dim_v1[3];
    }

    N         = dim_vi[3];
    v         = mxGetPr(mxCreateNumericArray(4,dim_vi,mxDOUBLE_CLASS,mxREAL));
    v_bias    = mxGetPr(mxGetField(model,0,"v_bias"));

    Hstride   = stride[0];
    Wstride   = stride[1];
    Hfilter   = s_filter[0];
    Wfilter   = s_filter[1];
    H         = dim_vi[0];
    W         = dim_vi[1];
    Hres      = dim_hi[0];
    Wres      = dim_hi[1];

    int offset_h  = (H-1)*Hstride+Hfilter-Hres;
    int offset_w  = (W-1)*Wstride+Wfilter-Wres;
    int H_off     = Hres + offset_h;
    int W_off     = Wres + offset_w;


    dim_offset    = (mwSize*)mxMalloc(sizeof(mwSize)*4);
    dim_offset[0] = H_off; dim_offset[1] = W_off; dim_offset[2] = n_map_h; dim_offset[3] = dim_vi[3];
    h_state_off   = mxGetPr(mxCreateNumericArray(4,dim_offset,mxDOUBLE_CLASS,mxREAL));

    for(ni = 0; ni < N; ni++){
        for(nh = 0; nh < n_map_h; nh++){
            for(j = 0; j < Wres; j++){
                for(i = 0; i < Hres; i++){
                    h_state_off[i+offset_h/2+H_off*(j+offset_w/2)+H_off*W_off*nh+H_off*W_off*n_map_h*ni]
                            = h_state[i+Hres*j+Hres*Wres*nh+Hres*Wres*n_map_h*ni];
                }
            }
        }
    }


    for(ni = 0; ni < N; ni++){
        for(nv = 0; nv < n_map_v; nv++){
            for (j = 0; j< W; j++){
                for (i = 0; i< H; i++){
                    id = i + H*j + H*W*nv + H*W*n_map_v*ni;
                    v[id] = 0;

                    for (nh = 0; nh < n_map_h; nh++){
                        for (jj = 0; jj < Wfilter; jj ++){
                            for (ii = 0; ii < Hfilter; ii++){

                                v[id] += h_state_off[(i*Hstride+ii)+H_off*(j*Wstride+jj)+H_off*W_off*nh+H_off*W_off*n_map_h*ni]
                                        * weights[Hfilter*Wfilter-1-(ii+Hfilter*jj)+Hfilter*Wfilter*nv+Hfilter*Wfilter*n_map_v*nh];
                            }
                        }
                    }

                    v_input[id] = v[id] +  v_bias[nv];

                    /* If the input type is not binary: v_sample[id] = v_input[id]; */
                    /*If the input is binary:v_sample[id] = 1.0/(1.0+exp(-v_input[id])); */
                    if (type[0] == 'B')
                        v_sample[id] = 1.0/(1.0+exp(-v_input[id]));
                    if (type[0] == 'G')
                        v_sample[id] = v_input[id];


                }
            }
        }
    }

    mxFree(dim_vi);
    mxFree(dim_offset);
	return;
}


