function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                  #1 - STEP 1 - CLUSTER ASSIGNMENT
%% Assigning each training example x(i) to its closest centroid
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m=size(X, 1);

% iterating through all training examples
for i=1:m;
    
    % ith training example vector
    x = X(i,:);

    % vector taking all distances between x and centroid
    distances = zeros(K, 1);
    
    % iterating over all centroid to compute the distance between x and centroid c_i
    for c_i=1:K;
        distances(c_i) = (x - centroids(c_i,:)) * (x - centroids(c_i,:))';
    end

    % We want to smallest distance and store the index in idx(i)
    [value, idx(i)] = min(distances);
end


% =============================================================

end

