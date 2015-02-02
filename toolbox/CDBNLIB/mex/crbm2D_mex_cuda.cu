#include <math.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Matlab - mex
#include <mex.h>
#include <matrix.h>

// CUDA
#include <cuda_runtime_api.h>


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#define dbg_print mexPrintf

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template<class T>
class CRBM_Data
{
public:
    T       *data_input;                // input data
    T       *data_kernel;               // kernel
    T       *h_bias;                    // bias of hidden layer
    T       *v_bias;                    // bias of visible layer
    T       *h_sample;                  // hidden values of h_sample
    T       *h_sample_init;             // initialization of hidden layer
    T       *h_state;                   // the active matrix
    T       *v_sample;                  // the visible layer values
    T        gauss;                     // gaussian parameter

    int     H, W;                       // input image H & W
    int     N;                          // image number
    int     Wfilter, Hfilter;           // kernel W & H
    int     Wres, Hres;                 // output data W & H
    int     Hstride, Wstride;           // stride of H & W
    int     Hpool, Wpool;               // pool size of H & W
    int     n_map_v, n_map_h;           // map number of v & h
    char    type_input;                 // type of inputdata

    int     run_on_gpu;                 // run on GPU (1, default) or CPU (0)

public:
    CRBM_Data(void) {
        run_on_gpu = 1;
        init();
    }

    ~CRBM_Data(void) {
        if( run_on_gpu )
            release();
        else
            release_no_free();
    }

    int init(void) {
        data_input    = NULL;
        data_kernel   = NULL;
        h_bias        = NULL;
        v_bias        = NULL;
        h_sample      = NULL;
        h_sample_init = NULL;
        h_state       = NULL;
        v_sample      = NULL;

        return 0;
    }

    int release(void) {
        if( data_input    != NULL ) delete [] data_input;
        if( data_kernel   != NULL ) delete [] data_kernel;
        if( h_bias        != NULL ) delete [] h_bias;
        if( v_bias        != NULL ) delete [] v_bias;
        if( h_sample      != NULL ) delete [] h_sample;
        if( h_sample_init != NULL ) delete [] h_sample_init;
        if( h_state       != NULL ) delete [] h_state;
        if( v_sample      != NULL ) delete [] v_sample;

        data_input    = NULL;
        data_kernel   = NULL;
        h_bias        = NULL;
        v_bias        = NULL;
        h_sample      = NULL;
        h_sample_init = NULL;
        h_state       = NULL;
        v_sample      = NULL;

    }

    int release_no_free(void) {
        data_input    = NULL;
        data_kernel   = NULL;
        h_bias        = NULL;
        v_bias        = NULL;
        h_sample      = NULL;
        h_sample_init = NULL;
        h_state       = NULL;
        v_sample      = NULL;
    }


    // NOTE: this function must be called after parameters have been set.
    void set_data_input(double *di) {
        if( !run_on_gpu ) {
            // FIXME: not safe
            if( sizeof(double) == sizeof(T) ) {
                data_input = (float*)di;
            } else {
                dbg_print("ERR: input data type is wrong! please input double type!\n");
            }

            return;
        }

        int n = H*W*n_map_v*N;
        if( data_input == NULL ) data_input = new T[n];

        for(int i=0; i<n; i++) data_input[i] = float(di[i]);
    }


    // NOTE: this function must be called after parameters have been set.
    // FIXME: only call by GPU type
    void get_data_input(double **di) {
        int n = H*W*n_map_v*N;
        for(int i=0; i<n; i++) di[i] = data_input[i];
    }



    // NOTE: this function must be called after parameters have been set.
    void set_data_kernel(double *di) {
        if( !run_on_gpu ) {
            // FIXME: not safe
            if( sizeof(double) == sizeof(T) ) {
                data_kernel = (float*)di;
            } else {
                dbg_print("ERR: input data type is wrong! please input double type!\n");
            }

            return;
        }

        int n = Hfilter*Wfilter*n_map_v*n_map_h;
        if( data_kernel == NULL ) data_kernel = new T[n];

        for(int i=0; i<n; i++) data_kernel[i] = float(di[i]);
    }


    // NOTE: this function must be called after parameters have been set.
    // FIXME: only call by GPU type
    void get_data_kernel(double **di) {
        int n = Hfilter*Wfilter*n_map_v*n_map_h;
        for(int i=0; i<n; i++) di[i] = data_kernel[i];
    }


};

/*** ------------------CUDA CONVOLUTION INFERENCE------------------------- ***/

__global__ void conv_cuda_infer(float *da, float *db, float *dc, int H, int W,
                                int Hres, int Wres, int Hfilter,int Wfilter,
                                int Hstride, int Wstride, int Hpool, int Wpool,
                                int n_map_v, int n_map_h,
                                int ni)
{
    int vmap_idx = blockIdx.x, hmap_idx = blockIdx.y;
    int conv_xi  = threadIdx.x, conv_yi;
    int ii, jj;

    float   *da_, *db_, *dc_;
    float   sum;

    // debug
    //arr_calc[vmap_idx*n_map_h + hmap_idx] = 1;

    // get array pointers
    da_ = da + ni*H*W*n_map_v + H*W*vmap_idx;       // input data
    db_ = db + Hfilter*Wfilter*n_map_v*hmap_idx +   // conv kernel
               Hfilter*Wfilter*vmap_idx;
    dc_ = dc + ni*Hres*Wres*n_map_h*n_map_v +       // output data
               Hres*Wres*n_map_v*hmap_idx +
               Hres*Wres*vmap_idx;

    // begin calculation
    for(conv_yi=0; conv_yi<Hres; conv_yi++) {
        sum = 0;

        for(jj =0; jj < Wfilter; jj++) {
            for(ii = 0; ii<Hfilter; ii++) {
                sum += da_[conv_yi*Hstride+ii + H*(conv_xi*Wstride+jj)]
                        * db_[ii + jj*Hfilter];
            }
        }

        dc_[conv_yi+Hres*conv_xi] = sum;
    }
}



/*** ------------------CUDA CONVOLUTION RECONSTRUCTION--------------------- ***/

__global__ void conv_cuda_recon(float *da, float *db, float *dc, int H_off, int W_off,
                                int H, int W, int Hfilter,int Wfilter,
                                int Hstride, int Wstride, int Hpool, int Wpool,
                                int n_map_v, int n_map_h,
                                int ni)
{
    int hmap_idx = blockIdx.x, vmap_idx = blockIdx.y;
    int conv_xi  = threadIdx.x, conv_yi;
    int ii, jj;

    float   *da_, *db_, *dc_;
    float   sum;


    // get array pointers
    da_ = da + ni*H_off*W_off*n_map_h + H_off*W_off*hmap_idx;       // input data
    db_ = db + Hfilter*Wfilter*n_map_v*hmap_idx +   // conv kernel
               Hfilter*Wfilter*vmap_idx;
    dc_ = dc + ni*H*W*n_map_v*n_map_h +       // output data
               H*W*n_map_h*vmap_idx +
               H*W*hmap_idx;

    // begin calculation
    for(conv_yi=0; conv_yi<H; conv_yi++) {
        sum = 0;

        for(jj =0; jj < Wfilter; jj++) {
            for(ii = 0; ii<Hfilter; ii++) {
                sum += da_[conv_yi*Hstride+ii + H_off*(conv_xi*Wstride+jj)]
                        * db_[Hfilter*Wfilter-1-(ii + jj*Hfilter)];
            }
        }

        dc_[conv_yi+H*conv_xi] = sum;
    }
}


/*** -------------------------CUDA MERGE INFERENCE------------------------- ***/

__global__ void conv_merge_infer(float *dc, float *dh, float *dd, int H, int W,
                                 int Hres, int Wres,int Hfilter,int Wfilter,
                                 int Hstride, int Wstride, int Hpool, int Wpool,
                                 int n_map_v, int n_map_h,
                                 int ni, char type_input, float gauss)
{
    int hmap_idx = blockIdx.x, vmap_idx;
    int jj,ii;

    float   *dc_, *dd_;


    dd_ = dd + ni*Hres*Wres*n_map_h + Hres*Wres*hmap_idx;

    // merge maps to single feature map
    for(vmap_idx = 0; vmap_idx < n_map_v; vmap_idx++) {
        dc_ = dc + ni*Hres*Wres*n_map_h*n_map_v + Hres*Wres*n_map_v*hmap_idx + Hres*Wres*vmap_idx;

        for(jj = 0; jj < Wres; jj++) {
            for(ii = 0; ii < Hres; ii++) {
                dd_[ii+jj*Hres] += dc_[ii+jj*Hres];
            }
        }
    }

    // apply bias
    for(jj = 0; jj < Wres; jj++) {
        for(ii = 0; ii < Hres; ii++) {

            if (type_input == 'B')
                dd_[ii+jj*Hres] = exp(dd_[ii+jj*Hres] + dh[hmap_idx]);
            if (type_input == 'G')
                dd_[ii+jj*Hres] = exp(1.0/(gauss*gauss)*(dd_[ii+jj*Hres] + dh[hmap_idx]));
        }
    }
}



/*** -------------------------CUDA MERGE RECONSTRUCTION---------------------- ***/

__global__ void conv_merge_recon(float *dc, float *dv, float *dd, int H_off, int W_off,
                                 int H, int W,int Hfilter,int Wfilter,
                                 int Hstride, int Wstride, int Hpool, int Wpool,
                                 int n_map_v, int n_map_h,
                                 int ni, char type_input)
{
    int vmap_idx = blockIdx.x, hmap_idx;
    int jj,ii;

    float   *dc_, *dd_;


    dd_ = dd + ni*H*W*n_map_v + H*W*vmap_idx;

    // merge maps to single feature map
    for(hmap_idx = 0; hmap_idx < n_map_h; hmap_idx++) {
        dc_ = dc + ni*H*W*n_map_v*n_map_h + H*W*n_map_h*vmap_idx + H*W*hmap_idx;

        for(jj = 0; jj < W; jj++) {
            for(ii = 0; ii < H; ii++) {
                dd_[ii+jj*H] += dc_[ii+jj*H];
            }
        }
    }

    // apply bias
    for(jj = 0; jj < W; jj++) {
        for(ii = 0; ii < H; ii++) {

            if (type_input == 'B')
                dd_[ii+jj*H] = 1.0/(1.0+exp(-(dd_[ii+jj*H] + dv[vmap_idx])));
            if (type_input == 'G')
                dd_[ii+jj*H] = dd_[ii+jj*H] + dv[vmap_idx];
        }
    }
}


//BOTTOM-UP: POSITIVE UPDATE
void crbm_inference2D(CRBM_Data<float> *p)
{
    int         ni, i, j, ii, jj, nh, nv, id,
                H, W, n_map_v, n_map_h, N,
                Hfilter, Wfilter, Hstride, Wstride,
                Hpool, Wpool, Hres, Wres;

    int         *_id;

    float       sum, rnd, pro_sum, gauss;
    float       *block;
    bool        done;
    char        type_input;

    H       = p->H;
    W       = p->W;
    N       = p->N;
    Hres    = p->Hres;
    Wres    = p->Wres;
    Hpool   = p->Hpool;
    Wpool   = p->Wpool;
    n_map_v = p->n_map_v;
    n_map_h = p->n_map_h;
    Hfilter = p->Hfilter;
    Wfilter = p->Wfilter;
    Hstride = p->Hstride;
    Wstride = p->Wstride;
    gauss   = p->gauss;
    type_input = p->type_input;


    // Initialize matrixs
    j       = Hres*Wres*n_map_h*N;
    block   = new float[j];
    for(i= 0; i< j; i++) block[i] = 0;

    _id     = new int[Hpool*Wpool];
    for(i= 0; i< Hpool*Wpool; i++) _id[i] = 0;

    /***---------------------------CUDA CODE------------------------------***/

    int SIZE_IMAGE, SIZE_FILTER, SIZE_OUTPUT;
    float *da, *db, *dc, *dd, *dh, *fc;

    j = Hres*Wres*n_map_v*n_map_h*N;
    fc = new float[j];
    for(i=0; i< j; i++) fc[i] = 0;
    //cudaMallocHost(&fc, sizeof(float)*Hres*Wres*n_map_v*n_map_h*N);
    //memset(fc, 0, sizeof(float)*Hres*Wres*n_map_v*n_map_h*N);

    SIZE_IMAGE  = H * W * n_map_v * N;
    SIZE_FILTER = Hfilter * Wfilter * n_map_v * n_map_h;
    SIZE_OUTPUT = Hres * Wres * n_map_h * N;

    cudaMalloc(&da, sizeof(float) * SIZE_IMAGE);
    cudaMalloc(&db, sizeof(float) * SIZE_FILTER);
    cudaMalloc(&dc, sizeof(float) * SIZE_OUTPUT*n_map_v);
    cudaMalloc(&dd, sizeof(float) * SIZE_OUTPUT);
    cudaMalloc(&dh, sizeof(float) * n_map_h);

    cudaMemcpy(da,p->data_input,    sizeof(float)*SIZE_IMAGE,         cudaMemcpyHostToDevice);
    cudaMemcpy(db,p->data_kernel,   sizeof(float)*SIZE_FILTER,        cudaMemcpyHostToDevice);
    cudaMemcpy(dc,fc,               sizeof(float)*SIZE_OUTPUT*n_map_v,cudaMemcpyHostToDevice);
    cudaMemcpy(dd,block,            sizeof(float)*SIZE_OUTPUT        ,cudaMemcpyHostToDevice);
    cudaMemcpy(dh,p->h_bias,        sizeof(float)*n_map_h,            cudaMemcpyHostToDevice);

    dim3    blocks(n_map_v, n_map_h);
    dim3    threads(Wres, 1);

    dim3    blocks2(n_map_h, 1);
    dim3    threads2(1, 1);

    for(ni=0; ni< N; ni++){

        conv_cuda_infer<<<blocks, threads>>>(da, db, dc,
                                             H, W, Hres, Wres, Hfilter, Wfilter,
                                             Hstride, Wstride, Hpool, Wpool,
                                             n_map_v,n_map_h, ni);

        conv_merge_infer<<<blocks2, threads2>>>(dc,dh, dd,
                                             H, W, Hres, Wres, Hfilter, Wfilter,
                                             Hstride, Wstride, Hpool, Wpool,
                                             n_map_v, n_map_h, ni, type_input, gauss);
    }

    cudaMemcpy(block, dd, sizeof(float) * SIZE_OUTPUT, cudaMemcpyDeviceToHost);


    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cudaFree(dd);
    cudaFree(dh);
    //cudaFreeHost(fc);
    delete [] fc;


    /***---------------------------CUDA END------------------------------***/

    /*** CONVOLUTION & GET HIDDEN ACTIVATION STATE ***/
    for(ni=0; ni< N; ni++){
        for(nh=0; nh< n_map_h; nh++){

            //GET HIDDEN ACTIVATION STATE

            for(j=0; j< floor(Wres/Wpool); j++){
                for(i=0; i< floor(Hres/Hpool); i++){

                    sum = 0;

                    for(jj=0; jj< Wpool; jj++){

                        _id[jj*Hpool] = i*Hpool + (j*Wpool+jj)*Hres + Hres*Wres*nh + Hres*Wres*n_map_h*ni;
                        sum          += block[_id[jj*Hpool]];

                        for(ii=1; ii< Hpool; ii++){
                            _id[jj*Hpool+ii] = _id[jj*Hpool+ii-1] + 1;
                            sum             += block[_id[jj*Hpool+ii]];
                        }
                    }

                    done     = false;
                    rnd      = rand() % 10000 / 10000.0;
                    pro_sum  = 0.0;

                    for(jj=0; jj< Hpool*Wpool; jj++){
                        p->h_sample[_id[jj]] = block[_id[jj]]/(1.0+sum);
                        pro_sum += p->h_sample[_id[jj]];

                        //Randomly generate the hidden state: at most one unit is activated
                        if(done == false){
                            if(pro_sum >= rnd){
                                p->h_state[_id[jj]] = 1;
                                done = true;
                            }
                        }
                    }

                }
            }

        }
    }

    delete [] _id;
    delete [] block;
    return;
}

// UP-DOWN: NEGATIVE UPDATE
void crbm_reconstruct2D(CRBM_Data<float> *p)
{
    int         ni, i, j, ii, jj, nh, nv, id,
                H, W, n_map_v, n_map_h, N,
                Hfilter, Wfilter, Hstride, Wstride,
                Hpool, Wpool, Hres, Wres,
                offset_h, offset_w, H_off, W_off;

    float       *h_state_off, *v;
    char        type_input;

    H       = p->H;
    W       = p->W;
    N       = p->N;
    Hres    = p->Hres;
    Wres    = p->Wres;
    Hpool   = p->Hpool;
    Wpool   = p->Wpool;
    n_map_v = p->n_map_v;
    n_map_h = p->n_map_h;
    Hfilter = p->Hfilter;
    Wfilter = p->Wfilter;
    Hstride = p->Hstride;
    Wstride = p->Wstride;
    type_input = p->type_input;

    j = H*W*n_map_v*N;
    v = new float[j];
    for(i=0; i< j; i++) v[i] = 0;

    //extend the matrix of h_state
    offset_h  = (H-1)*Hstride*Hstride+(Hfilter-1)*Hstride+Hfilter-H;
    offset_w  = (W-1)*Wstride*Wstride+(Wfilter-1)*Wstride+Wfilter-W;
    H_off     = Hres + offset_h;
    W_off     = Wres + offset_w;

    j           = H_off*W_off*n_map_h*N;
    h_state_off = new float[j];
    for(i=0; i< j; i++) h_state_off[i] = 0;

    for(ni=0; ni< N; ni++){
        for(nh=0; nh< n_map_h; nh++){
            for(j=0; j< Wres; j++){
                for(i=0; i< Hres; i++){

                    h_state_off[i + offset_h/2 + H_off*(j+offset_w/2) + H_off*W_off*nh + H_off*W_off*n_map_h*ni]
                   = p->h_state[i + Hres*j + Hres*Wres*nh + Hres*Wres*n_map_h*ni];

                }
            }
        }
    }

    /***--------------------------CUDA CODE----------------------------***/
    if (0) {
    int SIZE_IMAGE, SIZE_FILTER, SIZE_OUTPUT;
    float *da, *db, *dc, *dd, *dv, *fc;

    j = H*W*n_map_h*n_map_v*N;
    fc = new float[j];
    for(i=0; i< j; i++) fc[i] = 0;

    SIZE_IMAGE  = H_off * W_off * n_map_h * N;
    SIZE_FILTER = Hfilter * Wfilter * n_map_v * n_map_h;
    SIZE_OUTPUT = H * W * n_map_v * N;

    cudaMalloc(&da, sizeof(float) * SIZE_IMAGE);
    cudaMalloc(&db, sizeof(float) * SIZE_FILTER);
    cudaMalloc(&dc, sizeof(float) * SIZE_OUTPUT*n_map_h);
    cudaMalloc(&dd, sizeof(float) * SIZE_OUTPUT);
    cudaMalloc(&dv, sizeof(float) * n_map_h);

    cudaMemcpy(da,h_state_off,    sizeof(float)*SIZE_IMAGE,         cudaMemcpyHostToDevice);
    cudaMemcpy(db,p->data_kernel, sizeof(float)*SIZE_FILTER,        cudaMemcpyHostToDevice);
    cudaMemcpy(dc,fc,             sizeof(float)*SIZE_OUTPUT*n_map_v,cudaMemcpyHostToDevice);
    cudaMemcpy(dd,v,              sizeof(float)*SIZE_OUTPUT        ,cudaMemcpyHostToDevice);
    cudaMemcpy(dv,p->v_bias,      sizeof(float)*n_map_h,            cudaMemcpyHostToDevice);

    dim3    blocks(n_map_h, n_map_v);
    dim3    threads(W, 1);

    dim3    blocks2(n_map_v, 1);
    dim3    threads2(1, 1);

    for(ni=0; ni< N; ni++){

        conv_cuda_recon<<<blocks, threads>>>(da, db, dc,
                                             H_off, W_off, H, W, Hfilter, Wfilter,
                                             Hstride, Wstride, Hpool, Wpool,
                                             n_map_v,n_map_h, ni);

        conv_merge_recon<<<blocks2, threads2>>>(dc,dv, dd,
                                             H_off, W_off, H, W, Hfilter, Wfilter,
                                             Hstride,Wstride,Hpool,Wpool,
                                             n_map_v,n_map_h, ni,type_input);
    }

    cudaMemcpy(v, dd, sizeof(float) * SIZE_OUTPUT, cudaMemcpyDeviceToHost);

    for(i=0; i< H*W*n_map_v*N; i++) p->v_sample[i] = v[i];
    }

    /***---------------------------CUDA END---------------------------***/


    //do the convolution
    for(ni=0; ni< N; ni++){
        for(nv=0; nv< n_map_v; nv++){
            for(j=0; j< W; j++){
                for(i=0; i< H; i++){

                    id    = i + H*j + H*W*nv + H*W*n_map_v*ni;
                    v[id] = 0;

                    for (nh = 0; nh< n_map_h; nh++){
                        for (jj = 0; jj< Wfilter; jj++){
                            for (ii = 0; ii < Hfilter; ii++){

                                v[id] += h_state_off[(i*Hstride+ii) + H_off*(j*Wstride+jj) + H_off*W_off*nh + H_off*W_off*n_map_h*ni]
                                        * p->data_kernel[Hfilter*Wfilter-1-(ii+Hfilter*jj) + Hfilter*Wfilter*nv + Hfilter*Wfilter*n_map_v*nh];

                            }

                        }
                    }

                    v[id]          += p->v_bias[nv];

                    if (type_input == 'B')
                        p->v_sample[id] = 1.0/(1.0+exp(-v[id]));
                    if (type_input == 'G')
                        p->v_sample[id] = v[id];

                }
            }
        }
    }


    delete [] h_state_off;
    delete [] v;
    //delete [] fc;
    return;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    /***------------------LOAD DATA FROM MATLAB-------------------***/
    const mxArray       *model, *layer, *batch_data;
    double              *data_input, *data_kernel,
                        *s_filter, *stride, *pool,
                        *v_bias, *h_bias, *gaussian;

    mwSize              *dim_v, *dim_h;
    int                 i, j, ii, jj, ni, nv, nh,
                        Hfilter, Wfilter, Hstride, Wstride,
                        Hpool, Wpool, H, W, Hres, Wres,
                        n_map_v, n_map_h, ndim_v, N, id;

    mxChar              *type;


    //Check the structure of the fisrt input data
    if(mxIsStruct(prhs[0]) & mxIsStruct(prhs[1])){
        model       = prhs[0];
        layer       = prhs[1];
        batch_data  = prhs[2];
    }
    else{
        mexErrMsgTxt("You must specify two structure arrays!");
    }

    gaussian    = mxGetPr(mxGetField(model,0,"start_gau"));
    type        = mxGetChars(mxGetField(layer,0,"type_input"));
    data_input  = mxGetPr(batch_data);
    data_kernel = mxGetPr(mxGetField(model,0,"W"));
    s_filter    = mxGetPr(mxGetField(layer,0,"s_filter"));
    stride      = mxGetPr(mxGetField(layer,0,"stride"));
    pool        = mxGetPr(mxGetField(layer,0,"s_pool"));
    n_map_v     = mxGetScalar(mxGetField(layer,0,"n_map_v"));
    n_map_h     = mxGetScalar(mxGetField(layer,0,"n_map_h"));
    dim_v       = (mwSize*)mxGetDimensions(mxGetField(model,0,"v_input"));
    dim_h       = (mwSize*)mxGetDimensions(mxGetField(model,0,"h_input"));
    ndim_v      = mxGetNumberOfDimensions(mxGetField(model,0,"v_input"));
    v_bias      = mxGetPr(mxGetField(model,0,"v_bias"));
    h_bias      = mxGetPr(mxGetField(model,0,"h_bias"));

    // check the number of images
    if (ndim_v == 4)
        N = dim_v[3];
    else
        N = 1;


    CRBM_Data<float> crbm_data;

    crbm_data.Hfilter     = int(s_filter[0]);
    crbm_data.Wfilter     = int(s_filter[1]);
    crbm_data.Hstride     = int(stride[0]);
    crbm_data.Wstride     = int(stride[1]);
    crbm_data.n_map_v     = n_map_v;
    crbm_data.n_map_h     = n_map_h;
    crbm_data.Hpool       = int(pool[0]);
    crbm_data.Wpool       = int(pool[1]);
    crbm_data.H           = int(dim_v[0]);
    crbm_data.W           = int(dim_v[1]);
    crbm_data.Hres        = int(dim_h[0]);
    crbm_data.Wres        = int(dim_h[1]);
    crbm_data.N           = N;
    crbm_data.type_input  = type[0];
    crbm_data.gauss       = gaussian[0];


    Hfilter     = int(s_filter[0]);
    Wfilter     = int(s_filter[1]);
    Hstride     = int(stride[0]);
    Wstride     = int(stride[1]);
    Hpool       = int(pool[0]);
    Wpool       = int(pool[1]);
    H           = int(dim_v[0]);
    W           = int(dim_v[1]);
    Hres        = int(dim_h[0]);
    Wres        = int(dim_h[1]);

    // convert mex data to inner data
    //crbm_data.set_data_input(&data_input);
    //crbm_data.set_data_kernel(&data_kernel);

    j = H*W*n_map_v*N;
    crbm_data.data_input = new float[j];
    for(i=0; i< j; i++) crbm_data.data_input[i] = data_input[i];

    j = Hfilter*Wfilter*n_map_v*n_map_h;
    crbm_data.data_kernel = new float[j];
    for(i=0; i< j; i++) crbm_data.data_kernel[i] = data_kernel[i];


    // h_sample, h_sample_init, h_state
    j = crbm_data.Hres*crbm_data.Wres*n_map_h*N;
    crbm_data.h_sample_init = new float[j];
    crbm_data.h_sample      = new float[j];
    crbm_data.h_state       = new float[j];
    for(i =0 ; i < j; i++){
        crbm_data.h_sample_init[i] = 0;
        crbm_data.h_sample[i]   = 0;
        crbm_data.h_state[i]    = 0;
    }

    // v_sample
    j = crbm_data.H*crbm_data.W*n_map_v*N;
    crbm_data.v_sample = new float[j];
    for(i=0; i< j; i++) crbm_data.v_sample[i] = 0;
    // h_bias
    crbm_data.h_bias = new float[n_map_h];
    for(i=0; i< n_map_h; i++) crbm_data.h_bias[i] = float(h_bias[i]);
    // v_bias
    crbm_data.v_bias = new float[n_map_v];
    for(i=0; i< n_map_v; i++) crbm_data.v_bias[i] = float(v_bias[i]);



    /***------------------ GIBBS SAMPLE------------------------ ***/

    cudaSetDevice(0); // with gpu

    crbm_inference2D(&crbm_data);

    j = crbm_data.Hres*crbm_data.Wres*n_map_h*N;
    for(i=0; i< j; i++) crbm_data.h_sample_init[i] = crbm_data.h_sample[i];

    crbm_reconstruct2D(&crbm_data);

    j = crbm_data.H*crbm_data.W*n_map_v*N;
    for(i=0; i< j; i++) crbm_data.data_input[i] = crbm_data.v_sample[i];

    crbm_inference2D(&crbm_data);


    /***----------------CALCULATE DW---------------------------***/
    double    *dW;
    mxArray   *dW_array;
    mwSize    *dim_w;

    dim_w     = (mwSize*)mxGetDimensions(mxGetField(model,0,"W"));
    dW_array  = mxCreateNumericArray(4,dim_w,mxDOUBLE_CLASS, mxREAL);
    dW        = mxGetPr(dW_array);


    for(ni = 0; ni < N; ni++){

        for(nh = 0; nh < n_map_h; nh++){
            for (j = 0; j < Wfilter; j++){
                for (i = 0; i < Hfilter; i++){

                    for(nv = 0; nv < n_map_v; nv++){
                        for(jj = 0; jj < Wres; jj++){
                            for (ii = 0; ii < Hres; ii++){

                                id      = i + Hfilter*j + Hfilter*Wfilter*nv + Hfilter*Wfilter*n_map_v*nh;
                                dW[id] += (data_input[(ii*Hstride+i) + H*(jj*Wstride+j) + H*W*nv+H*W*n_map_v*ni]
                                        * crbm_data.h_sample_init[(ii+Hres*jj) + Hres*Wres*nh + Hres*Wres*n_map_h*ni]
                                        - crbm_data.v_sample[(ii*Hstride+i) + H*(jj*Wstride+j) + H*W*nv+H*W*n_map_v*ni]
                                        * crbm_data.h_sample[(ii+Hres*jj) + Hres*Wres*nh + Hres*Wres*n_map_h*ni]);
                            }
                        }
                    }
                }
            }
        }
    }

    /*-------RETURN: (h_sample, h_sample_init, v_sample, dW) TO MATLAB----------*/

    mxArray        *h_sample_array, *h_sam_in_array, *v_sample_array;
    mwSize         *dim_hi, *dim_vi;
    double         *h_sample, *h_sample_init, *v_sample;

    dim_hi         = (mwSize*)mxMalloc(sizeof(mwSize)*4);
    dim_vi         = (mwSize*)mxMalloc(sizeof(mwSize)*4);
    dim_hi[0] = Hres; dim_hi[1] = Wres; dim_hi[2] = n_map_h; dim_hi[3] = N;
    dim_vi[0] = H;    dim_vi[1] = W;    dim_vi[2] = n_map_v; dim_vi[3] = N;

    h_sample_array = mxCreateNumericArray(4,dim_hi,mxDOUBLE_CLASS, mxREAL);
    h_sample       = mxGetPr(h_sample_array);
    h_sam_in_array = mxCreateNumericArray(4,dim_hi,mxDOUBLE_CLASS, mxREAL);
    h_sample_init  = mxGetPr(h_sam_in_array);
    v_sample_array = mxCreateNumericArray(4,dim_vi,mxDOUBLE_CLASS, mxREAL);
    v_sample       = mxGetPr(v_sample_array);

    // set the values to mex type matrix
    j = Hres*Wres*n_map_h*N;
    for(i=0; i< j; i++){
        h_sample_init[i] = crbm_data.h_sample_init[i];
        h_sample[i]      = crbm_data.h_sample[i];
    }

    j = H*W*n_map_v*N;
    for(i=0; i< j; i++) v_sample[i] = crbm_data.v_sample[i];


    const char *fieldname[] = {"h_sample","h_sample_init","v_sample","dW"};
    mxArray    *struct_array;
    struct_array = plhs[0] = mxCreateStructMatrix(1,1,4,fieldname);

    mxSetField(struct_array,0,"h_sample",h_sample_array);
    mxSetField(struct_array,0,"h_sample_init",h_sam_in_array);
    mxSetField(struct_array,0,"v_sample",v_sample_array);
    mxSetField(struct_array,0,"dW",dW_array);

    mxFree(dim_vi);
    mxFree(dim_hi);


}




