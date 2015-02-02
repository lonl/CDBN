%------------------------------------------------------------------------------%
%               This is a 2D BINARY convolutional Deep Belief Networks
%------------------------------------------------------------------------------%

clear all;

% SET DEMO PARAMETERS 
demo_add_noise = 0;


%% ------------------------------ LOAD DATA --------------------------------- %%

%LOAD MNIST DATA TO TEST THE BINARY CDBN

load ./data/mnist/mnistSmall.mat;
train_data     = trainData;
train_data     = reshape(train_data', [28,28,1,10000]);
test_data      = testData;
test_data      = reshape(test_data', [28,28,1,2000]);

train_data     = train_data(:,:,:,1:2000);
test_data      = test_data(:,:,:,1:2000);
trainL         = trainLabels(1:2000,:);
testL          = testLabels(1:2000,:);

% ADD NOISE
if demo_add_noise
    fprintf('------------------- ADD NOISE IN TEST DATA ------------------- \n');
    b          = rand(size(test_data)) > 0.9;
    noised     = test_data;
    rnd        = rand(size(test_data));
    noised(b)  = rnd(b);
    test_data  = noised;
end


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
layer{1}.whiten         = 0;
layer{1}.type_input     = 'Binary'; % OR 'Gaussian' 'Binary'

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
layer{2}.whiten         = 0;
layer{2}.type_input     = 'Binary';

% THIRD LAYER SETTING ...   % YOU CAN CONTINUE TO SET THE THIRD, FOURTH,
                            % FIFTH LAYERS' PARAMETERS WITH THE SAME STRUCT
                            % MENTIONED ABOVE


%% ----------- GO TO 2D CONVOLUTIONAL DEEP BELIEF NETWORKS ------------------ %% 
tic;

[model,layer] = cdbn2D(layer);
save('./model/model_parameter','model','layer');

toc;

trainD  = model{1}.output;
trainD1 = model{2}.output;


%% ------------ TESTDATA FORWARD MODEL WITH THE PARAMETERS ------------------ %%
% FORWARD MODEL OF NETWORKS
H = length(layer);
layer{1}.inputdata = test_data;
fprintf('output the testdata features:>>...\n');

tic;
if H >= 2
    
    % PREPROCESSS INPUTDATA TO BE SUITABLE FOR TRAIN 
    layer{1} = preprocess_train_data2D(layer{1});
    model{1}.output = crbm_forward2D_batch_mex(model{1},layer{1},layer{1}.inputdata);
    
    for i = 2:H
        layer{i}.inputdata = model{i-1}.output;
        layer{i} = preprocess_train_data2D(layer{i});
        model{i}.output = crbm_forward2D_batch_mex(model{i},layer{i},layer{i}.inputdata);
    end
    
else
    
    layer{1} = preprocess_train_data2D(layer{1});
    model{1}.output = crbm_forward2D_batch_mex(model{1},layer{1},layer{1}.inputdata);
end

testD  = model{1}.output;
testD1 = model{2}.output;
toc;

%% ------------------------------- Softmax ---------------------------------- %%

fprintf('train the softmax:>>...\n');

tic;

% TRANSLATE THE OUTPUT TO ONE VECTOR
trainDa = [];
trainLa = [];
for i= 1:size(trainD,4)
    a1 = [];
    a2 = [];
    a3 = [];
    for j = 1:size(trainD,3)
        a1 = [a1;reshape(trainD(:,:,j,i),size(trainD,2)*size(trainD,1),1)];
    end
    
    for j = 1:size(trainD1,3)
        a2 = [a2;reshape(trainD1(:,:,j,i),size(trainD1,2)*size(trainD1,1),1)];
    end
    a3 = [a3;a1;a2];
    trainDa = [trainDa,a3];
    trainLa = [trainLa;find(trainL(i,:)==1)];
end

testDa = [];
testLa = [];
for i= 1:size(testD,4)
    b1 = [];
    b2 = [];
    b3 = [];
    for j = 1:size(testD,3)
        b1 = [b1;reshape(testD(:,:,j,i),size(testD,2)*size(testD,1),1)];
    end
    
    for j =1:size(testD1,3)
        b2 = [b2;reshape(testD1(:,:,j,i),size(testD1,2)*size(testD1,1),1)];
    end
    b3 = [b3;b1;b2];
    testDa = [testDa,b3];
    testLa = [testLa;find(testL(i,:)==1)];
end

%save('./model/model_out','trainDa','trainLa','testDa','testLa');

% TRAIN THE CLASSIFIER & TEST THE TESTDATA
softmaxExercise(trainDa,trainLa,testDa,testLa);
toc;


%% ------------------------------- Figure ----------------------------------- %%

%  POOLING MAPS
figure(1);
[r,c,n] = size(model{1}.output(:,:,:,1));
visWeights(reshape(model{1}.output(:,:,:,1),r*c,n)); colormap gray
title(sprintf('The first Pooling output'))
drawnow

% ORIGINAL SAMPLE
figure(2);
imagesc(layer{1}.inputdata(:,:,:,1)); colormap gray; axis image; axis off
title(sprintf('Original Sample'));




