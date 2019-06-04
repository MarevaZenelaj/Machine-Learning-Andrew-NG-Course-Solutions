function [theta, J_history] = gradientDescent(X, y, theta, alpha, iterations)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(iterations, 1);

for iter = 1:iterations
    sumV = zeros(2,1);
    dv_J_ = zeros(2,1);
    for i = 1:m
      h_th  = theta(1) + theta(2)*X(i,2);
      sumV(1) = sumV(1) + (h_th-y(i)); 
      sumV(2) = sumV(2) + (h_th-y(i))*X(i,2);
    end
    
    dv_J_ = 1/m * sumV;
    theta  = theta - alpha * dv_J_;
    J_history(iter) = computeCost(X, y, theta);
    
end
end
