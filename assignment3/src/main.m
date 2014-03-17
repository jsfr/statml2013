%%
% III.1.1
%
clear;fprintf('############### III.1.1 ###############\n');

trainData = importdata('../data/sincTrain25.dt');
testData = importdata('../data/sincValidate10.dt');

h = @(a) a / (1 + abs(a));
hdiff = @(a) 1 / (1 + abs(a))^2;

[deltaInWeights, deltaOutWeights] = backPropagation(trainData, h, hdiff, [1 1; 1 1], [1 1 1]')

[numDeltaInWeights, numDeltaOutWeights] = numericalDiffs(trainData, h, [1 1; 1 1], [1 1 1]', 10E-8)

%%
% III.1.2
%
clear;fprintf('############### III.1.2 ###############\n');

trainData = importdata('../data/sincTrain25.dt');
testData = importdata('../data/sincValidate10.dt');

h = @(a) a / (1 + abs(a));
hdiff = @(a) 1 / (1 + abs(a))^2;

StartInWeights = ones(20,2);
StartOutWeights = ones(21, 1);

[InWeights, OutWeights] = steepestDescent(trainData, [1 1; 1 1], [1 1 1]', h, hdiff, 10E-4, 0.01)
[InWeights, OutWeights] = steepestDescent(trainData, StartInWeights, StartOutWeights, h, hdiff, 10E-4, 0.01)

trainError = meanSquaredError(trainData, InWeights, OutWeights, h)
testError = meanSquaredError(testData, InWeights, OutWeights, h)

%%
% III.2.1
%
% clear;fprintf('############### III.2.1 ###############\n');

% trainData = importdata('../data/parkinsonsTrainStatML.dt');
% testData = importdata('../data/parkinsonsTestStatML.dt');

% means = mean(trainData(:, 1:end-1), 1)'
% vars = var(trainData(:, 1:end-1), 0, 1)'
% stds = std(trainData(:, 1:end-1), 0, 1)';

% normTrainData = [ fNorm(trainData(:, 1:end-1), means', stds') trainData(:, end) ];
% normTestData = [ fNorm(testData(:, 1:end-1), means', stds') testData(:, end) ];

% normTrainMeans =  mean(normTrainData(:, 1:end-1), 1)'
% normTainVars = var(normTrainData(:, 1:end-1), 0, 1)'

% normTestMeans =  mean(normTestData(:, 1:end-1), 1)'
% normTestVars = var(normTestData(:, 1:end-1), 0, 1)'