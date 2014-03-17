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

%[InWeights, OutWeights] = steepestDescent(trainData, [1 1; 1 1], [1 1 1]', h, hdiff, 10E-4, 0.01)
%[InWeights, OutWeights] = steepestDescent(trainData, StartInWeights, StartOutWeights, h, hdiff, 10E-4, 0.01)

%trainError = meanSquaredError(trainData, InWeights, OutWeights, h)
%testError = meanSquaredError(testData, InWeights, OutWeights, h)


smallLearningRate = 0.001;
largeLearningRate = 0.1;

[InWeights, OutWeights, TrainError2nodesLargeRate, TestError2nodesLargeRate] = steepestDescentPlot(trainData, testData, [1 1; 1 1], [1 1 1]', h, hdiff, 10E-4, largeLearningRate);
[InWeights, OutWeights, TrainError2nodesSmallRate, TestError2nodesSmallRate] = steepestDescentPlot(trainData, testData, [1 1; 1 1], [1 1 1]', h, hdiff, 10E-4, smallLearningRate);

[InWeights, OutWeights, TrainError20nodesLargeRate, TestError20nodesLargeRate] = steepestDescentPlot(trainData, testData, StartInWeights, StartOutWeights, h, hdiff, 10E-4, largeLearningRate);
[InWeights, OutWeights, TrainError20nodesSmallRate, TestError20nodesSmallRate] = steepestDescentPlot(trainData, testData, StartInWeights, StartOutWeights, h, hdiff, 10E-4, smallLearningRate);

h = figure(1);
semilogy(TrainError2nodesSmallRate(:,:), 'b-', 'LineWidth', 1.5);
hold on;
semilogy(TestError2nodesSmallRate(:,:), 'b--', 'LineWidth', 1.5);

semilogy(TrainError20nodesSmallRate(:,:), 'r-', 'LineWidth', 1.5);
semilogy(TestError20nodesSmallRate(:,:), 'r--', 'LineWidth', 1.5);

semilogy(TrainError2nodesLargeRate(:,:), 'g-', 'LineWidth', 1.5);
semilogy(TestError2nodesLargeRate(:,:), 'g--', 'LineWidth', 1.5);

semilogy(TrainError20nodesLargeRate(:,:), 'c-', 'LineWidth', 1.5);
semilogy(TestError20nodesLargeRate(:,:), 'c--', 'LineWidth', 1.5);

legend('Training data 2 hidden nodes small rate', 'Test data 2 hidden nodes small rate', ...
	'Training data 20 hidden nodes small rate', 'Test data 20 hidden nodes small rate', ...
	'Training data 2 hidden nodes large rate', 'Test data 2 hidden nodes large rate', ...
	'Training data 20 hidden nodes large rate', 'Test data 20 hidden nodes large rate');
ylabel('MSE');
xlabel('Learning epoch');
betterPlots(h);
print(h, '-depsc2', '../figures/III12_1.eps');
hold off;




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