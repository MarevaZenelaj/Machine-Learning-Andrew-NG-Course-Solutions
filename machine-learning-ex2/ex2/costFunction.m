function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
sumV = 0.0;
sumT = 0.0;
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cohst of a particular choice of teta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
h_theta = zeros(size(X,1));
h_theta = sigmoid(theta'*X');

for i = 1:m
  sumV = sumV + y(i)*log(h_theta(i))+(1-y(i))*log(1-h_theta(i));
 end
J = -1/m*sumV;



for t = 1:size(theta)
  sumT = 0.0;
  for k = 1:m
  sumT = sumT + (h_theta(k) - y(k))*(X(k,t));
  end
  grad(t) = 1/m*sumT;
end
% =============================================================

end
