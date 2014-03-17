function [deltaInWeights, deltaOutWeights] = numericalDiffs(Data, h, InWeights, OutWeights, epsilon)
    ys = arrayfun(@(x) neuralNetwork(h, x, InWeights, OutWeights), Data(:,1));
    E = sum((ys - Data(:, 2)).^2) / size(Data, 1);

    e = eye(size(InWeights, 1));
    deltaInWeights = zeros(size(InWeights, 1), 1);
    for j = 1:size(InWeights, 1)
        InWeightsEpsilon = InWeights + repmat(epsilon*e(:,j),1,size(InWeights,2));
        ys = arrayfun(@(x) neuralNetwork(h, x, InWeightsEpsilon, OutWeights), Data(:,1));
        Eepsilon = sum((ys - Data(:, 2)).^2) / size(Data, 1);
        deltaInWeights(j,:) = (Eepsilon - E) / epsilon;
    end
    deltaInWeights = 1/2 * deltaInWeights;

    e = eye(size(OutWeights, 1));
    deltaOutWeights = zeros(size(OutWeights, 1), 1);
    for j = 1:size(OutWeights, 1)
        OutWeightsEpsilon = OutWeights + repmat(epsilon*e(:,j),1,size(OutWeights,2));
        ys = arrayfun(@(x) neuralNetwork(h, x, InWeights, OutWeightsEpsilon), Data(:,1));
        Eepsilon = sum((ys - Data(:, 2)).^2) / size(Data, 1);
        deltaOutWeights(j,:) = (Eepsilon - E) / epsilon;
    end
    deltaOutWeights = 1/2 * deltaOutWeights;
end