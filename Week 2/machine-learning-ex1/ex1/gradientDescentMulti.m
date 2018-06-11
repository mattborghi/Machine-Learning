function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    temp0 = 0;
    temp1 = 0;
    temp2 = 0;
    for k=1:length(theta)
        partial = zeros(m,1);
        spartial = 0;
        for i = 1:m
            for j=1:length(theta)
                partial(i,1) = partial(i,1) + theta(j,1)*X(i,j);
            end
            partial(i,1) = ( partial(i,1)-y(i) )*X(i,k);
            spartial = spartial + partial(i,1);
        end

        if k == 1 
            temp0 = theta(1) - alpha*spartial/m;
        elseif k==2     
            temp1 = theta(2) - alpha*spartial/m;
        else 
            temp2 = theta(3) - alpha*spartial/m;
        end   
    end
    
    %Update the theta values
    theta(1) = temp0;
    theta(2) = temp1;
    theta(3) = temp2;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    %fprintf('For iteration %d, J(theta) = %f\n',iter,J_history(iter));

end

end
