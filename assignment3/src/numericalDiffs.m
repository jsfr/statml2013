function [deltaInWeights, deltaOutWeights] = numericalDiffs(Data, h, InWeights, OutWeigths)
    epsilon = 10E-8;
    ys = arrayfun(@(x) neuralNetwork(h, x, InWeights, OutWeights), Data(1,:));
    E = sum((ys - Data(:, 2)).^2) / size(Data, 1);

    for 
end