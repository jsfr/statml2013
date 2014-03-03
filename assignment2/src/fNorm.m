function normData = fNorm(data)
    means = mean(data(:,1:end-1), 1);
    stds = std(data(:,1:end-1), 0, 1);
    normData = data(:,1:end-1) - repmat(means, size(data,1), 1);
    normData = normData ./ repmat(stds, size(data,1), 1);
    normData = [ normData data(:, end) ];
end