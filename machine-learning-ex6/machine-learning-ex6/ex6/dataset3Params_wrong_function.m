
setC = [0.01 0.03 0.1 0.3 1 3 10 30];
setS = [0.01 0.03 0.1 0.3 1 3 10 30];
error = zeros(64,1);

for i = 1:length(setC),
  C = setC(i);
  for j = 1:length(setS),
    model = svmTrain(X, y, setC(i), @(x1, x2) gaussianKernel(x1, x2, setS(j)));
    sigma = setS(j);
    predictions = svmPredict(model, Xval);
    error(i+j-1) = mean(double(predictions ~= yval));
  endfor
endfor

[M, I] = min(error);

C = ceil(I/length(setC));
sigma = mod(I,length(setC));


% ======