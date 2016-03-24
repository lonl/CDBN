function model = crbm2D(layer)
%------------------------------------------------------------------------------%
%                  2D convolutional Restricted Boltzmann Machine               %
% INPUT:                                                                       %
%        <LAYER>: - Parameters in each layer                                   %
% OUTPUT:                                                                      %
%        <MODEL>: - Weights & Bias & ... in each layer                         %
%------------------------------------------------------------------------------%


%% --------------------- INITIALIZE THE PARAMETERS IN MODEL ----------------- %%

cpu               = layer.cpu;
batchsize         = layer.batchsize;
model.n_cd        = layer.n_cd;
model.momentum    = layer.momentum;
model.start_gau   = layer.start_gau;
model.stop_gau    = layer.stop_gau;
model.beginAnneal = Inf;
layer.s_inputdata = [size(layer.inputdata,1),size(layer.inputdata,2)];

% INITIALIZE THE WEIGHTS
model.W           = 0.01*randn(layer.s_filter(1), layer.s_filter(2), layer.n_map_v, layer.n_map_h);
model.dW          = zeros(size(model.W));
model.v_bias      = zeros(layer.n_map_v, 1);
model.dV_bias     = zeros(layer.n_map_v, 1);
model.h_bias      = zeros(layer.n_map_h, 1);
model.dH_bias     = zeros(layer.n_map_h, 1);
model.v_size      = [layer.s_inputdata(1), layer.s_inputdata(2)];
model.v_input     = zeros(layer.s_inputdata(1), layer.s_inputdata(2), layer.n_map_v,batchsize);          
model.h_size      = (layer.s_inputdata - layer.s_filter)./(layer.stride) + 1;
model.h_input     = zeros(model.h_size(1), model.h_size(2), layer.n_map_h, batchsize);
model.error       = zeros(layer.n_epoch,1);


% ADD SOME OTHER PARAMETERS FOR TEST
model.winc = 0;
model.hinc = 0;
model.vinc = 0;

% NEED TO FIX THE STRIDE HERE
H_out             = ((size(layer.inputdata,1)-layer.s_filter(1))/layer.stride(1)+1)/layer.s_pool(1);
W_out             = ((size(layer.inputdata,2)-layer.s_filter(2))/layer.stride(2)+1)/layer.s_pool(2);
model.output      = zeros([H_out, W_out,layer.n_map_h,size(layer.inputdata,4)]);

% CREATING BATCH DATA
N                 = size(layer.inputdata,4);
numcases          = size(layer.inputdata,4);
numbatches        = ceil(N/batchsize);
groups            = repmat(1:numbatches, 1, batchsize);
groups            = groups(1:N);
perm              = randperm(N);
groups            = groups(perm);
dW                = zeros(size(model.dW));

for i = 1:numbatches
    batchdata{i}  = layer.inputdata(:,:,:,groups == i);
end

%% ------------------------------ TRAIN THE CRBM ---------------------------- %%

for epoch = 1:layer.n_epoch
    err = 0;
    sparsity = zeros(1,numbatches);
   
    tic;
    % FOR EACH EPOCH, ALL SAMPLES ARE COMPUTED
    for i = 1:numbatches
        
        batch_data  = batchdata{i};
        
        
        %-----------------------------------------------------------------%
        switch cpu
            case 'mex'
            %----------------- HERE COMPLETE WITH MEX FUNCTION -----------%
                [model_new] = crbm2D_batch_mex(model,layer,batch_data);

                dW                  = model_new.dW;
                model.v_sample      = model_new.v_sample;
                model.h_sample      = model_new.h_sample;
                model.h_sample_init = model_new.h_sample_init;
        
            case 'cuda'
                %-------------- HERE COMPLETE WITH CUDA FUNCTION ---------%
                [model_new] = crbm2D_mex_cuda(model,layer,batch_data);

                dW                  = model_new.dW;
                model.v_sample      = model_new.v_sample;
                model.h_sample      = model_new.h_sample;
                model.h_sample_init = model_new.h_sample_init;        
        
            case 'matlab'
                %-------------- HERE COMPLETE WITH MATLAB FUNCTION -------%
                [model, dW] = calc_gradient2D(model,layer,batch_data);
        end % switch
        
        
        %-----------------------------------------------------------------%
        
        % UPDATE THE MODEL PARAMETERS
        model = apply_gradient2D(model,layer,batch_data,dW);
        
        sparsity(i) = mean(model.h_sample_init(:)); 
        
        err1        = (batch_data - model.v_sample).^2;
        err         = err + sum(sum(sum(sum(err1))));
    end % numbatches
    

    if (model.start_gau > model.stop_gau)
        model.start_gau = model.start_gau*0.99;
    end
        
    model.error(epoch) = err;
    
    fprintf('    epoch %d/%d, reconstruction err %f, sparsity %f\n', epoch,layer.n_epoch,err, mean(sparsity(:)));
    toc;
  
end

%% ----------------------- OUTPUT THE POOLING LAYER ------------------------- %%

% CHOOSE DIFFERENT FORWARD MODEL
switch cpu
    case 'matlab'
        output = crbm_forward2D(model,layer,layer.inputdata);
        model.output = output;
    case 'mex'
        output = crbm_forward2D_batch_mex(model,layer,layer.inputdata);
        model.output = output;
    case 'cuda'
        output = crbm_forward2D_batch_mex(model,layer,layer.inputdata);
        model.output = output;
end % switch

end % function


