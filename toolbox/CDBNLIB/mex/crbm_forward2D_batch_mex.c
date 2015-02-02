#include <math.h>
#include <mex.h>
#include <matrix.h>
#include <time.h>
#include <string.h>

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	const mxArray  *model, *layer, *batch_data;
	      mxArray  *model_new, *h_input_array, *h_sample_array, *output_array;
    int            ni,N,n_dim,n_map_h, n_map_v, nh,nv,j,i,jj,ii,id,
	               Hstride,Wstride,Hfilter,Wfilter,Hres,Wres,H,W,Hpool,Wpool,Hout,Wout;
	int            *_id;
	double         *s_filter, *stride, *h, *data, *weights, *h_bias, *block, *pool,
                   *h_input, *h_sample, *output, *gaussian;
	mwSize         *dim_vi, *dim_hi, *dim_id, *dim_h, *dim_out;
    mxChar         *type;

    model          = prhs[0];
	layer          = prhs[1];
	batch_data     = prhs[2];
	
    dim_vi         = mxGetDimensions(batch_data);
    n_dim          = mxGetNumberOfDimensions(batch_data);

    if (n_dim == 2 || n_dim == 3)
    	N = 1;
    else
    	N = dim_vi[3];

    dim_h          = (mwSize*)mxMalloc(sizeof(mwSize)*4);
	dim_hi         = mxGetDimensions(mxGetField(model,0,"h_input"));
    dim_h[0]       = dim_hi[0];
    dim_h[1]       = dim_hi[1];
    dim_h[2]       = dim_hi[2];
    dim_h[3]       = N;

    n_map_h        = mxGetScalar(mxGetField(layer,0,"n_map_h"));
	n_map_v        = mxGetScalar(mxGetField(layer,0,"n_map_v"));
    s_filter       = mxGetPr(mxGetField(layer,0,"s_filter"));   
    stride         = mxGetPr(mxGetField(layer,0,"stride"));
    data           = mxGetPr(batch_data);
    weights        = mxGetPr(mxGetField(model,0,"W"));
    h              = mxGetPr(mxCreateNumericArray(4,dim_h,mxDOUBLE_CLASS,mxREAL));
    h_bias         = mxGetPr(mxGetField(model,0,"h_bias"));
    block          = mxGetPr(mxCreateNumericArray(4,dim_h,mxDOUBLE_CLASS,mxREAL));
    pool           = mxGetPr(mxGetField(layer,0,"s_pool"));
	h_input_array  = mxCreateNumericArray(4,dim_h,mxDOUBLE_CLASS, mxREAL);
	h_input        = mxGetPr(h_input_array);
	h_sample_array = mxCreateNumericArray(4,dim_h,mxDOUBLE_CLASS, mxREAL);
	h_sample       = mxGetPr(h_sample_array);
    dim_out        = mxGetDimensions(mxGetField(model,0,"output"));
    dim_out[3]     = N;
    output_array   = plhs[0] = mxCreateNumericArray(4,dim_out,mxDOUBLE_CLASS, mxREAL);
    output         = mxGetPr(output_array);
    gaussian       = mxGetPr(mxGetField(model,0,"start_gau"));
    type           = mxGetChars(mxGetField(layer,0,"type_input"));

    /*Here need to pay attention to the _id:mxUINT32_CLASS*/
    dim_id         = (mwSize*)mxMalloc(sizeof(mwSize)*2);
    dim_id[0]      = pool[0]; dim_id[1] = pool[1];
    _id            = mxGetPr(mxCreateNumericArray(2,dim_id,mxUINT32_CLASS,mxREAL));

    mxFree(dim_id);
    mxFree(dim_h);

    Hstride        = stride[0];
    Wstride        = stride[1];
    Hfilter        = s_filter[0];
    Wfilter        = s_filter[1];
    H              = dim_vi[0];
    W              = dim_vi[1];
    Hres           = dim_hi[0];
    Wres           = dim_hi[1];
    Hpool          = pool[0];
    Wpool          = pool[1];
    Hout           = floor(Hres/Hpool);
    Wout           = floor(Wres/Wpool);

    for (ni = 0; ni < N; ni++){
        for (nh = 0; nh < n_map_h; nh++){

            for (j = 0; j < Wres; j++){
                for (i = 0; i < Hres; i++){
                    id = i+Hres*j+Hres*Wres*nh+Hres*Wres*n_map_h*ni;
                    h[id] = 0;
                    h_input[id] = 0;

                    for (nv = 0; nv < n_map_v; nv++){
                        for (jj = 0; jj < Wfilter; jj++){
                            for (ii = 0; ii < Hfilter; ii++){

                                h[id] += data[(i*Hstride+ii)+H*(j*Wstride+jj)+H*W*nv+H*W*n_map_v*ni]
                                        * weights[(ii+Hfilter*jj)+Hfilter*Wfilter*nv+Hfilter*Wfilter*n_map_v*nh];

                            }
                        }
                    }

                    h_input[id] = h[id] + h_bias[nh];

                    /* for crbm blocksum & outpooing */
                    if (type[0] == 'B')
                        block[id] = exp(h_input[id]);
                    if (type[0] == 'G')
                        block[id] = exp(1.0/(gaussian[0]*gaussian[0])*h_input[id]);
                }
            }


            /* output the pooling & crbm blocksum: hidden activation summation */

            for (j = 0; j < Wout; j++){
                for (i = 0; i < Hout; i++){

                    double sum = 0.0;
                    for (jj = 0; jj < Wpool; jj++){
                        _id[jj*Hpool] = i*Hpool+(j*Wpool+jj)*Hres + Hres*Wres*nh+Hres*Wres*n_map_h*ni;
                        sum += block[_id[jj*Hpool]];
                        for (ii = 1; ii < Hpool; ii++){

                            _id[jj*Hpool+ii] = _id[jj*Hpool+ii-1] + 1;
                            sum += block[_id[jj*Hpool+ii]];

                        }
                    }

                    int out_id = i+j*Hout+Hout*Wout*nh+Hout*Wout*n_map_h*ni;
                    for (jj = 0; jj < Hpool*Wpool; jj++){
                        h_sample[_id[jj]] = 1.0-(1.0/(1.0+sum));

                    }

                    output[out_id] = h_sample[_id[0]];
                }
            }
        }
    }

    return;

}

