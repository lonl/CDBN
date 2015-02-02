%------------------------------------------------------------------------------%
%               This is a demo softmax for mnist data classification
%------------------------------------------------------------------------------%

clear all;

% SET DEMO PARAMETERS 
demo_add_noise = 0;

%% ------------------------------ LOAD DATA -----------------------------------%%

load ./data/mnist/mnistSmall.mat;

trainDa     = [];
trainLa     = [];
trainDa     = trainData';

for i       = 1:size(trainData,1)
    trainLa = [trainLa,find(trainLabels(i,:)==1)];
end

testDa      = [];
testLa      = [];
testDa      = testData';
for i       = 1:size(testData,1)
    testLa  = [testLa,find(testLabels(i,:)==1)];
end

trainDa     = trainDa(:,1:2000);
trainLa     = trainLa(:,1:2000);

% ADD NOISE
if demo_add_noise
    fprintf('------------------- ADD NOISE IN TEST DATA ------------------- \n');
    b         = rand(size(testDa)) > 0.9;
    noised    = testDa;
    r         = rand(size(testDa));
    noised(b) = r(b);
    testDa    = noised;
end

%% ------------------------------ DIRECTED SOFTMAX -----------------------------%%

softmaxExercise(trainDa,trainLa,testDa,testLa);
