function [model,dW] = calc_gradient2D(model,layer,data)
% ---------------------------------------------------------------------------- %
%                    caculate dW, dV_bias, dH_bias
% INPUT: 
%       <MODEL>: - Weights & Bias in each layer 
%       <LAYER>: - Parameters in each layer
%       <DATA>;  - Input data
%
% OUTPUT:
%       <MODEL>: - Weights & Bias in each layer 
%       <DW>   : - Delta weights
% ---------------------------------------------------------------------------- %

%% --------------------------------GIBBS SAMPLES -----------------------------%%

% CRBM INFERENCE
model = crbm_inference2D(model,layer,data);
%[model.h_sample, model.h_state] = crbm_inference2D_mex(model,layer,data); 

model.h_sample_init = model.h_sample;
for i = 1:model.n_cd
    
    % CRBM RECONSTRUCT
    model = crbm_reconstruct2D(model,layer);
    %[model.v_sample,hh] = crbm_reconstruct2D_mex(model,layer,model.h_state);
    
    
    % CRBM INFERENCE
    model = crbm_inference2D(model,layer,model.v_sample);
    %[model.h_sample,model.h_state] = crbm_inference2D_mex(model,layer,model.v_sample);
end

% CALCULATE THE DW
dW = zeros(size(model.W));
for i = 1 : layer.n_map_h
    for j = 1 : layer.n_map_v
        dW(:,:,j,i) = conv2(data(:,:,j,1),model.h_sample_init(end:-1:1,end:-1:1,i,1),'valid') ...
                    - conv2(model.v_sample(:,:,j,1),model.h_sample(end:-1:1,end:-1:1,i,1),'valid');
        
    end
end

end