function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
% ------------------
% size(X)->12 2
% size(y)-> 12 1
% size(theta)-> 2 1
% ------------------
% theta -> 1 1 initial values
% X -> 1's in first column

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

Jreg =  lambda/(2*m)*sum(theta(2:end,1).^2);
%Agregar theta0's = 1
%temp = verzcat( 1 , theta); % -> 3x1
J = sum( ( X*theta - y).^2 )/(2*m) + Jreg;


temp  = theta;
temp(1,1) = 0;

gradreg = ( ( X' * ( X*theta - y ) ) )/m; % j = 0

grad = gradreg + lambda*temp/m; % j >= 1


% =========================================================================

grad = grad(:);

end
