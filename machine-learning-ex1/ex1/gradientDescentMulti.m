function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
%features = size(X,2);


    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
 %for iter = 1:num_iters
  %  sumV = zeros(features,1);
   % dv_J_ = zeros(features,1);
    %for jk = 1:features
     % for i = 1:m
      %  h_th  = theta * X(i,:);
       % sumV(jk) = sumV(jk) + (h_th-y(i))*X(i,jk); 
      %end
    %end
    %dv_J_ = 1/m * sumV;
    %theta  = theta - alpha * dv_J_;
    

for iter = 1:num_iters
    theta = theta - alpha * (1/m) * (((X*theta) - y)' * X)'; % Vectorized  
    J_history(iter) = computeCostMulti(X, y, theta);
end

end
