%------------------------------------------------------------------------------%
%               This is a 2D GAUSSIAN convolutional Deep Belief Networks
%------------------------------------------------------------------------------%
clear all;

%% ------------------------------ LOAD DATA --------------------------------- %%

%% LOAD PICTURES TO TEST THE GAUSSIAN CDBN

for i = 1:9
    str = ['./data/MITcoast/image_000',num2str(i)];
    str1  = strcat(str,'.jpg');
    image = imread(str1);
    image = double(image)/255;
    train_data(:,:,i) = image;
end

train_data = reshape(train_data,[256,256,1,9]);
train_data = train_data(:,:,:,1:1);



%% ------------ INITIALIZE THE PARAMETERS OF THE NETWORK -------------------- %%

% FIRST LAYER SETTING
layer{1} = default_layer2D();  % DEFAULT PARAMETERS SETTING, 
                               % YOU CAN CHANGE THE PARAMETERS IN THE FOLLOWING LINES

layer{1}.inputdata      = train_data;
layer{1}.n_map_v        = 1;
layer{1}.n_map_h        = 9;
layer{1}.s_filter       = [7 7];
layer{1}.stride         = [1 1];  
layer{1}.s_pool         = [2 2];
layer{1}.n_epoch        = 10;
layer{1}.learning_rate  = 0.05;
layer{1}.sparsity       = 0.03;
layer{1}.lambda1        = 5;
layer{1}.lambda2        = 0.05;
layer{1}.whiten         = 1;
layer{1}.type_input     = 'Gaussian'; % OR 'Gaussian' 'Binary'

% SECOND LAYER SETTING
layer{2} = default_layer2D();  % DEFAULT PARAMETERS SETTING, 
                               % YOU CAN CHANGE THE PARAMETERS IN THE FOLLOWING LINES

layer{2}.n_map_v        = 9;
layer{2}.n_map_h        = 16;
layer{2}.s_filter       = [5 5];
layer{2}.stride         = [1 1];
layer{2}.s_pool         = [2 2];
layer{2}.n_epoch        = 10;
layer{2}.learning_rate  = 0.05;
layer{2}.sparsity       = 0.02;
layer{2}.lambda1        = 5;
layer{2}.lambda2        = 0.05;
layer{2}.whiten         = 1;
layer{2}.type_input     = 'Gaussian';

% THIRD LAYER SETTING ...   % YOU CAN CONTINUE TO SET THE THIRD, FOURTH,
                            % FIFTH LAYERS' PARAMETERS WITH THE SAME STRUCT
                            % MENTIONED ABOVE


%% ----------- GO TO 2D CONVOLUTIONAL DEEP BELIEF NETWORKS ------------------ %% 
tic;

[model,layer] = cdbn2D(layer);
save('./model/model_parameter','model','layer');

toc;

%% ------------------------------- Figure ----------------------------------- %%

%  THE FIRST LAYER POOLING MAPS
figure(1);
[r,c,n] = size(model{1}.output(:,:,:,1));
visWeights(reshape(model{1}.output(:,:,:,1),r*c,n)); colormap gray
title(sprintf('The first Pooling output'))
drawnow

%  THE SECOND LAYER POOLING MAPS
figure(2);
[r,c,n] = size(model{2}.output(:,:,:,1));
visWeights(reshape(model{2}.output(:,:,:,1),r*c,n)); colormap gray
title(sprintf('The second Pooling output'))
drawnow

% ORIGINAL SAMPLE
figure(3);
imagesc(layer{1}.inputdata(:,:,:,1)); colormap gray; axis image; axis off
title(sprintf('Original Sample'));




