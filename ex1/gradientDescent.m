function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %
    
    update = zeros(1, length(theta));
    
    for i=1:length(theta)
        update(i, :) = ((X * theta) - y) .* X(:, 1);
    end
    
    for i=1:length(theta)
        theta(i) = theta(i) - (alpha / m) * update(i);
    end
    
%     update1 = 0;
%     for i=1:m
%         update1 = update1 + (((X(i, :) * theta) - y(i)) * X(i, 1));
%     end
%     
%     update2 = 0;
%     for i=1:m
%         update2 = update2 + (((X(i, :) * theta) - y(i)) * X(i, 2));
%     end
%     
%     theta(1) = theta(1) - (alpha / m) * update1;
%     theta(2) = theta(2) - (alpha / m) * update2;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
