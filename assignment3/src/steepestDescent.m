function [InWeights, OutWeights, Errors] = steepestDescent(Data, InWeights, OutWeights, h, hdiff, epsilon, step)
    deltaError = epsilon + 1;
    oldError = meanSquaredError(Data, InWeights, OutWeights, h);
    elderError = oldError + 1;
    newError = elderError + 1;
    errorChange = 1;
    
    while (deltaError > epsilon) & errorChange
        [deltaInWeights, deltaOutWeights] = backPropagation(Data, h, hdiff, InWeights, OutWeights);
        deltaInWeights = step*sign(deltaInWeights); 
        deltaOutWeights = step*sign(deltaOutWeights);
        InWeights = InWeights - deltaInWeights;
        OutWeights = OutWeights - deltaOutWeights;
        newError = meanSquaredError(Data, InWeights, OutWeights, h);
        deltaError = abs(newError - oldError);
        errorChange = elderError ~= newError;
        elderError = oldError;
        oldError = newError;
    end
end