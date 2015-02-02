function model = crbm_reconstruct2D(model,layer)
%------------------------------------------------------------------------------%
%                            Top - down process
% INPUT: <MODLE>, <LAYER>
% OUTPUT: <MODEL>
%------------------------------------------------------------------------------%

h_state = double(rand(size(model.h_sample)) < model.h_sample);
model.v_input = zeros(size(model.v_input));
N = size(model.v_input,4);

for k = 1:N
    for i = 1:layer.n_map_v
        for j = 1:layer.n_map_h
            
            model.v_input(:,:,i,k) = model.v_input(:,:,i,k) + conv2(h_state(:,:,j,k),model.W(:,:,i,j),'full');
        end
        model.v_input(:,:,i,k) = model.v_input(:,:,i,k) + model.v_bias(i);
    end
end

if strcmp(layer.type_input, 'Binary')
    model.v_sample = sigmoid(model.v_input);
    
else
    model.v_sample = model.v_input;
    
end

end