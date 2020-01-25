function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% reshape([1, 2, 3, 4, 5, 3], 3, 2)
% ans = 1   4
%       2   5
%       3   3

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PART 1  - Feedforward and Cost Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1// FEEDFORWARD

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

%% 2// COST FUNCTION (unregularized)

% Based on y, which is a 5000x1 matrix , we recreate a 5000x10 matrix that takes
% 5000 rows of 10x1 vectors. We need to recode the labels as vectors containing only values 0 or 1
% In other words, I want to transalte my lowercase "y" vector (m x 1) into a matrix of 0's and 1's
% of size (m x 10)
% To do so, we start with a 10 x 10 Identity matrix that will be used the right sequence of numbers to
% the Y matrix. (eg: y(423) = 9, this transaltes into [0 0 0 0 0 0 0 0 1 0] vector, where the 1 is at the 
% ninth position)

% 10 x 10 Identity metrix => 10 because there are 10 digits 0,1,2,3,4,5,6...9
%
%   1 0 0 0 0 0 0 0 0 0
%   0 1 0 0 0 0 0 0 0 0
%   0 0 1 0 0 0 0 0 0 0
%   ...
% matrix to indicate the position of the number
I = eye(num_labels);

% m x 10 placehodler matrix of zeroes
% We will iterate over m-rows from the y vector (m x 1) and assign a row from the Identity matrix
% that corresponds to the digit.
Y = zeros(m, num_labels);

% Say that y = [5 2 2 1 2 4 5 1 2 3 4 5 5]
%
%   For i from 1 -> 5000:
%     i = 1:
%       y(1) = 5;
%       identity_row_at_index_y = I(5, :);
%       Y(1, :)  = identity_row_at_index_y; // here we assign 0 0 0 0 1 0 0 0 0 0 to matrix Y
%
% This results in a copy of y vector into a matrix
for i = 1:m
  Y(i, :) = I(y(i), :);
end

J = (1/m)*sum(sum((-Y).*log(h) - (1-Y).*log(1-h)));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PART 2 - BackProp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1// COST FUNCTION (regularized)

% new theta without zero index Theta
tmp_theta_1 = Theta1(:,2:end);
tmp_theta_2 = Theta2(:,2:end);

reg_term = lambda/(2*m) * (sum(sum(tmp_theta_1.^2)) + sum(sum(tmp_theta_2.^2)));

J = J + reg_term;

%% 2// COMPUTE DELTA's

d_3 = h - Y;
d_2 = (d_3 * Theta2)(:,2:end) .* sigmoidGradient(Z_2);

D_1 = d_2'*A_1;
D_2 = d_3'*A_2;

Theta1_grad = D_1./m + lambda*Theta1;
Theta2_grad = D_2./m + lambda*Theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
