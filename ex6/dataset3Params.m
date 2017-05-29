function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
Clist= [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]; 
sigmalist = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% Clist= [0.3];       % ultra-shortlist to save time on submission of later exercises
% sigmalist = [0.3];  % ditto
predictionError = zeros(length(Clist), length(sigmalist)); 
%
% loop for calculating mean error for each model
for i = 1:length(Clist)
    for j = 1:length(sigmalist)
        Ctest = Clist(1, i);
        sigmatest = sigmalist(1, j);
        model = svmTrain(X, y, Ctest, @(x1, x2) gaussianKernel(x1, x2, sigmatest)); 
        predictions = svmPredict(model, Xval);
        predictionError(i, j) = mean(double(predictions != yval));
        fprintf(['\nSVM for C = %f, sigma = %f, error = %f\n'],...
             Ctest, sigmatest, predictionError(i, j));
    end
end
% locate minimum prediction error
[minError, rmin] = min(predictionError); % min for each column, row index for each min
[minError, cmin] = min(minError);       % min of the mins, column index of the min of mins
rmin = rmin(cmin);                      % row index of the min of mins

C = Clist(rmin);
sigma = sigmalist(cmin);

% =========================================================================

end
