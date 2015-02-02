function [model,layer] = cdbn2D(layer)
%------------------------------------------------------------------------------%
%                    2D convolutional Deep Belief Networks                     %
% INPUT:                                                                       %
%      <LAYER>: - Parameters in each layer                                     %
% OUTPUT:                                                                      %
%      <MODEL>: - Weights & Bias in each layer                                 %
%      <LAYER>: - Parameters in each layer                                     %                
%------------------------------------------------------------------------------%


% NUMBER OF LAYERS
H = length(layer);

if H >= 2
    
    % TRAIN THE FIRST CRBM ON THE DATA
    layer{1} = preprocess_train_data2D(layer{1});
    model{1} = crbm2D(layer{1});
    
    % TRAIN ALL OTHER CRBMS ON TOP OF EACH OTHER
    for i = 2:H
        layer{i}.inputdata = model{i-1}.output;
        layer{i} = preprocess_train_data2D(layer{i});
        model{i} = crbm2D(layer{i});
    end
    
else
    
    % THE CDBN ONLY A SINGLE LAYER... BUT WE SHOULD WORK ANYWAY
    % TO TRAIN THE FIRST CRBM ON INPUTDATA
    layer{1} = preprocess_train_data2D(layer{1});
    model{1} = crbm2D(layer{1});
end


end
