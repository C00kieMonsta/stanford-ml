function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Why do we implement another algorithm for classifying handwritten numbers?  Is multi-class logistic regression not sufficient?
% Logistic regression cannot form more complex hypotheses as it is only a linear classiﬁer. You could add more features
% (such as polynomial features) to logistic regression, but that can be very expensive to train

% We have to add a vector of ones to X for the bias part x0

% Input layer
a0_1 = ones(m, 1);
A_1 = [a0_1, X];

% Hidden layer
a0_2 = ones(m, 1);
Z_2 = A_1 * Theta1';
A_2 = [a0_2, sigmoid(Z_2)];

% Output layer
Z_3 = A_2 * Theta2';
h = sigmoid(Z_3);

% We have to pick the class for which the corresponding logistic regression
% classifier outputs the highest probability
% this is a 5000 x 1 matrix, since we get the max val for each row
[values, indices] = max(h, [], 2);

% The predictions here is the index of the class, so we return the indices which corresponds to the class
p = indices;

% =========================================================================


end
