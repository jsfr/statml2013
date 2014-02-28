function delta = lda(trainData)
    C = unique(trainData(:, 3));
    [n m] = size(trainData); 
    
    mu = zeros(length(C), m);
    Pr = zeros(length(C), 1);
    for k = 1:length(C)
        Pr(k) = [ length(trainData(trainData(:, 3) == C(k), 3)) / length(n) C(k) ];
        mu(k,:) = mean(trainData(trainData(:, 3) == C(k), :));
    end

    Sigma = arrayfun(@(x, y, c) ([x y] - mu(mu(:,3) == c, 1:2))' * ([x y] - mu(mu(:,3) == c, 1:2)), trainData(:, 1), trainData(:, 2), trainData(:, 3), 'UniformOutput', false);
    Sigma = 1/(n - length(C)) * sum(cat(3,Sigma{:}),3);

    delta = cell(length(C), 2);
    for k = 1:length(C)
        delta{k} = { @(x) x * inv(Sigma) * mu(k,1:2)' - 1/2 mu(k,1:2) inv(Sigma) * mu(k,1:2)' + Pr(Pr(:, 2) == k, 1) C(k) };
    end

    getClass = @(x) cellfun(@(f,c) [ f(x) c ], delta{:,1}, delta{:,2})
end