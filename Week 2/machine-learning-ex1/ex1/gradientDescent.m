function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha
graphics_toolkit('gnuplot')
% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
index = zeros(num_iters,1);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    temp0 = 0;
    temp1 = 0;
    partial = 0;
    for i = 1:m
        partial = theta(1)+theta(2)*X(i,2)-y(i);
        temp0 = temp0 + partial;
        temp1 = temp1 + partial*X(i,2);
    end
    temp0 = theta(1) - alpha*temp0/m;
    temp1 = theta(2) - alpha*temp1/m;

    %Update the theta values
    theta(1) = temp0;
    theta(2) = temp1;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    index(iter) = iter;
    %fprintf('For iteration %d, J(theta) = %f\n',iter,J_history(iter));
end
    %plot(index,J_history,'MarkerSize',13);
    %xlabel('Number of iterations');
    %ylabel('J(theta)');
end
