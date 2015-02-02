function [layer] = preprocess_train_data2D(layer)
%%
% Here you should preprocess your code for pooling layer

mod_1 = mod((size(layer.inputdata,1)-layer.s_filter(1))/layer.stride(1)+1,layer.s_pool(1));
if mod_1~=0
   layer.inputdata(1:floor(mod_1/2),:,:,:) =[];
   layer.inputdata(end-ceil(mod_1/2)+1:end,:,:,:) =[];
end

mod_2 = mod((size(layer.inputdata,2)-layer.s_filter(2))/layer.stride(2)+1,layer.s_pool(2));
if mod_2~=0
   layer.inputdata(:,1:floor(mod_2/2),:,:) =[];
   layer.inputdata(:,end-ceil(mod_2/2)+1:end,:,:) =[];
end

if layer.whiten
if strcmp(layer.type_input, 'Gaussian')
    m = size(layer.inputdata,4);
    n = size(layer.inputdata,3);
    for i = 1 : m
        for j =1 : n
            layer.inputdata(:,:,j,i) = crbm_whiten(layer.inputdata(:,:,j,i));
        end
    end
    
end
end


function im_out = crbm_whiten(im)

if size(im,3)>1
    im = rgb2gray(im); 
end

im = double(im);
im = im - mean(im(:));
im = im./std(im(:));

N1 = size(im, 1);
N2 = size(im, 2);

[fx fy]=meshgrid(-N1/2:N1/2-1, -N2/2:N2/2-1);
rho=sqrt(fx.*fx+fy.*fy)';

f_0=0.4*mean([N1,N2]);
filt=rho.*exp(-(rho/f_0).^4);

If=fft2(im);
imw=real(ifft2(If.*fftshift(filt)));

im_out = 0.1*imw/std(imw(:)); % 0.1 is the same factor as in make-your-own-images

end

end
