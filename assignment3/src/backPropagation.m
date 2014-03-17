function [deltaInWeights, deltaOutWeights] = backPropagation(Data, h, hdiff, InWeights, OutWeights)
    noOfNodes = size(InWeights, 1);
    ys = arrayfun(@(x) neuralNetwork(h, x, InWeights, OutWeights), Data(:,1));
    deltaKs = ys - Data(:, 2);
    deltaInWeights = zeros(size(InWeights));
    deltaOutWeights = zeros(size(OutWeights'));
    for i=1:size(Data, 1)
        x = [1 Data(i,1)];
        a = arrayfun(@(k) sum(InWeights(k,:).*x), [1:noOfNodes]);
        z = [1 arrayfun(@(ai) h(ai), a)];
        deltaJs = arrayfun(@(w, ai) hdiff(ai)*w*deltaKs(i), OutWeights(2:end)', a);
        deltaInWeights = deltaInWeights + deltaJs' * x;
        deltaOutWeights = deltaOutWeights + deltaKs(i) * z;
    end
    % As we have mean squared error we need to divide by number of points
    deltaInWeights = deltaInWeights / size(Data, 1);
    deltaOutWeights = deltaOutWeights' / size(Data, 1);
end