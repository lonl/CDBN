function block = crbm_blocksum2D(model,layer)
%------------------------------------------------------------------------------%
%                        hidden activation summation
% INPUT: <MODEL>, <LAYER>
% OUTPUT: 
%        <BLOCK> - Summation of each pool region
%------------------------------------------------------------------------------%

if strcmp(layer.type_input, 'Binary')
    h_input = exp(model.h_input);
else
    h_input = exp(1.0/(model.start_gau^2).*model.h_input);
end

col = size(h_input,2);
row = size(h_input,1);
N   = size(h_input,4);
x_stride = layer.s_pool(2);
y_stride = layer.s_pool(1);
block = zeros(size(h_input));
for k = 1:N
for i = 1:floor(row/y_stride)
    offset_r = ((i-1)*y_stride+1):(i*y_stride);
    for j = 1:floor(col/x_stride)
        offset_c = ((j-1)*x_stride+1):(j*x_stride);
        block_val = squeeze(sum(sum(h_input(offset_r,offset_c,:,k))));
        block(offset_r,offset_c,:,k) = repmat(permute(block_val, [2,3,1]), numel(offset_r),numel(offset_c));
    end
end
end
end