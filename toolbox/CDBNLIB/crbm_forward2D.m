function output = crbm_forward2D(model,layer,data)
%------------------------------------------------------------------------------%
%                          calculate the output
% INPUT: <MODEL>, <LAYER>, <DATA>
% OUTPUT: <OUTPUT>
%------------------------------------------------------------------------------%


n = size(data,4);
x_stride = layer.s_pool(2);
y_stride = layer.s_pool(1);
row = size(model.h_sample,1);
col = size(model.h_sample,2);
output = zeros(floor(row/y_stride),floor(col/x_stride),layer.n_map_h,n);

for k = 1:n
    
    batch_data = data(:,:,:,k);
    model.h_input = zeros(row,col,size(model.h_sample,3),1);
	for i = 1:layer.n_map_h
        for j =1:layer.n_map_v
            model.h_input(:,:,i,1) = model.h_input(:,:,i,1) + conv2(batch_data(:,:,j,1),model.W(end:-1:1,end:-1:1,j,i),'valid');
        end
        model.h_input(:,:,i,1) = model.h_input(:,:,i,1) + model.h_bias(i);
    end
    
    block = crbm_blocksum2D(model,layer);
    h_sample = 1-(1./(1+block));
    output(:,:,:,k) = h_sample(1:y_stride:row, 1:x_stride:col,:);
    
end

        
end