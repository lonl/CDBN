function model = apply_gradient2D(model,layer,data,dW)
%------------------------------------------------------------------------------%
%                               Apply Gradient
% INPUT: <MODEL>, <LAYER>, <DATA>, <DW>
% OUTPUT: <MODEL>
%------------------------------------------------------------------------------%

N       = size(data,4);
dV_bias = squeeze(sum(sum(sum(data - model.v_sample,1),2),4));
dH_bias = squeeze(sum(sum(sum(model.h_sample_init - model.h_sample,1),2),4));

% USING THIS WAY WITH BINARY
if strcmp(layer.type_input, 'Binary')
    
    dW            = dW/N;
    dH_bias       = dH_bias/N;
    dV_bias       = dV_bias/N;
    
    model.dW      = model.momentum*model.dW + (1-model.momentum)*dW;
    model.W       = model.W + layer.learning_rate*(model.dW - layer.lambda2*model.W);
    
    penalty       = 0;
    model.dV_bias = model.momentum*model.dV_bias + (1-model.momentum)*dV_bias;
    model.v_bias  = model.v_bias + layer.learning_rate*(model.dV_bias - penalty*model.v_bias);
    
    model.dH_bias = model.momentum*model.dH_bias + (1-model.momentum)*dH_bias;
    model.h_bias  = model.h_bias + layer.learning_rate*(model.dH_bias - penalty*model.h_bias);
    
    model.h_bias  = model.h_bias + layer.learning_rate*layer.lambda1*...
        (squeeze(layer.sparsity-mean(mean(mean(model.h_sample_init,1),2),4)));
end

% TRY ANOTHER WAY TO UPDATE THE PARAMERTERS: GAUSSIAN
if strcmp(layer.type_input, 'Gaussian')
    
    N            = size(model.h_sample_init,1) * size(model.h_sample_init,2) * layer.batchsize;
    dW           = dW/N - 0.01*model.W;
    %dh           = (squeeze(sum(sum(model.h_sample_init,1),2)) - squeeze(sum(sum(model.h_sample,1),2)))/N - 0.05*(squeeze(mean(mean(model.h_sample_init,1),2)) - 0.002);
    dh           = (squeeze(sum(sum(sum(model.h_sample_init,1),2),4)) - squeeze(sum(sum(sum(model.h_sample,1),2),4)))/N;
    dv           = 0;
    
    model.winc   = model.winc*model.momentum + layer.learning_rate*dW;
    model.W      = model.winc + model.W;
    
    model.hinc   = model.hinc*model.momentum + layer.learning_rate*dh;
    model.h_bias = model.hinc + model.h_bias;
    
    model.vinc   = model.vinc*model.momentum + layer.learning_rate*dv;
    model.v_bias = model.vinc + model.v_bias;
end


end