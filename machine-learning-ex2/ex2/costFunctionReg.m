function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
sumV = 0.0;
sumT = 0.0;
sum_Theta = 0.0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_theta = zeros(size(X,1));
h_theta = sigmoid(theta'*X');

for i = 1:m
  sumV = sumV + y(i)*log(h_theta(i))+(1-y(i))*log(1-h_theta(i));
 end
 
for h = 2:size(theta) // theta 1 should not be considered in calculation 
  sum_Theta = sum_Theta + (theta(h))^2;
 end
 
J = -1/m*sumV + sum_Theta*lambda/(2*m);

sumT = 0.0;

for k = 1:m
  sumT = sumT + (h_theta(k) - y(k))*(X(k,1));
  end
grad(1) = 1/m*sumT;

for t = 2:size(theta)
  sumT = 0.0;
  for k = 1:m
  sumT = sumT + (h_theta(k) - y(k))*(X(k,t));
  end
  grad(t) = 1/m*sumT + lambda*theta(t)/m;


end 


% =============================================================

end