%------------------------------------------------------------------------------%
%               This is a 2D BINARY Deep Belief Networks
%------------------------------------------------------------------------------%

clear all;

% SET DEMO PARAMETERS 
demo_add_noise = 0;

%% ------------------------------LOAD DATA--------------------------------%%

load ./data/mnist/mnistSmall.mat;
data           = trainData;
testdata       = testData;

labels         = [];
for i          = 1 : size(trainLabels,1)
    labels     = [labels;find(trainLabels(i,:)==1)];
end

testlabels     = [];
for i          = 1 : size(testLabels,1)
    testlabels = [testlabels;find(testLabels(i,:)==1)];
end

data           = data(1:2000,:);
labels         = labels(1:2000,:);

% ADD NOISE
if demo_add_noise
    fprintf('------------------- ADD NOISE IN TEST DATA ------------------- \n');
    b          = rand(size(testdata)) > 0.9;
    noised     = testdata;
    r          = rand(size(testdata));
    noised(b)  = r(b);
    testdata   = noised;
end

%% ----------------------------- TWO LAYER DBN-----------------------------%%
op.verbose = true;
models     = dbnFit(data,[100 100],labels,op,op);
yhat2      = dbnPredict(models,testdata);

%% ------------------------------PRINT ACCURACY----------------------------%%
fprintf('Classification accuracy on testdata using DBN with 100-100 hiddens is %f\n', ...
    1.0-sum(yhat2~=testlabels)/length(yhat2));



