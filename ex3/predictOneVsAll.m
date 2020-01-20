function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% My classifiers have been trained, now I want to predict the class to which 

% X is m x n matrix = 5000 x 10
% all_theta is a n x num_labels matrix
% X * all_theta becomes a m x num_labels matrix
h = sigmoid(X*all_theta');

% We have to pick the class for which the corresponding logistic regression
% classifier outputs the highest probability
% this is a 5000 x 1 matrix, since we get the max val for each row
% Let is do it for the first row of what has been predicted with the logistic regression
%  >> row_1 = h(1, :)
%  >> [val, index] = max(row_1, [], 2)
% Running both commands above would give you the max among all values from the first row,
% as well as the index of where the value was found. Why index? To match the class it belongs to
% or here to simply match the number it predicted. If index is 10 and val is 97%, it means that
% the regression predicts that the written number should correspond do the number 10 with proba of 97%
[values, indices] = max(h, [], 2);

% The predictions here is the index of the class, so we return the indices which corresponds to the class
p = indices;


% =========================================================================


end
