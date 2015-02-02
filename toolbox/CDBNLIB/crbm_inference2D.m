function model = crbm_inference2D(model,layer,data)
%------------------------------------------------------------------------------%
%                             bottom-up process
% INPUT: <MODEL>, <LAYER> & <DATA>
% OUTPUT: <MODEL>
%------------------------------------------------------------------------------%

model.h_input = zeros(size(model.h_input));
N = size(model.h_input,4);
for k = 1:N
    for i = 1 : layer.n_map_h
        for j = 1 : layer.n_map_v
            model.h_input(:,:,i,k) = model.h_input(:,:,i,k) + conv2(data(:,:,j,k),model.W(end:-1:1,end:-1:1,j,i),'valid');
        end
        
        model.h_input(:,:,i,k) = model.h_input(:,:,i,k) + model.h_bias(i);
    end
end

%% ------------------------ hidden activation summation -----------------------%
block          = crbm_blocksum2D(model,layer);

if strcmp(layer.type_input, 'Binary')
    model.h_sample = exp(model.h_input)./(1+block);
else
    model.h_sample = exp(1.0/(model.start_gau^2).*model.h_input)./(1+block);
end

end