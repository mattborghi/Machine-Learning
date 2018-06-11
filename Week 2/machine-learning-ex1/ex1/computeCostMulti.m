function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%X matrix of m x 3 (2 of data 1 of 1s)
%y a column vector of m x 1
%theta a column vector of 3 x 1 (theta0 theta1 theta2)'
partial2 = zeros(m,1);
for i = 1:m
	for j=1:length(theta)
		partial2(i,1) = partial2(i,1) + theta(j,1)*X(i,j);
	end
	J = J + ( partial2(i,1) - y(i) )^2;
end

J = J / (2*m);
% =========================================================================

end
