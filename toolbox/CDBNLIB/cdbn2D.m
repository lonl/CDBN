function [model,layer] = cdbn2D(layer)
%------------------------------------------------------------------------------%
%                    2D convolutional Deep Belief Networks                     %
% INPUT:                                                                       %
%      <LAYER>: - Parameters in each layer                                     %
% OUTPUT:                                                                      %
%      <MODEL>: - Weights & Bias in each layer                                 %
%      <LAYER>: - Parameters in each layer                                     %                
%------------------------------------------------------------------------------%

    % TRAIN THE FIRST CRBM ON THE DATA
    layer{1} = preprocess_train_data2D(layer{1});
    fprintf('layer 1:\n');
    model{1} = crbm2D(layer{1});
    str1 = sprintf('./model/model_%s_%s_%d',layer{1}.type_input,layer{1}.cpu,1);
    save(str1,'model','layer');
    
    % TRAIN ALL OTHER CRBMS ON TOP OF EACH OTHER
    for i = 2:length(layer)
        fprintf('layer %d:\n',i);
        layer{i}.inputdata = model{i-1}.output;
        layer{i} = preprocess_train_data2D(layer{i});
        model{i} = crbm2D(layer{i});
        str1 = sprintf('./model/model_%s_%s_%d',layer{i}.type_input,layer{i}.cpu,i);
        save(str1,'model','layer');
    end

end
