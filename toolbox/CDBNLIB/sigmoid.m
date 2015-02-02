function [ y ] = sigmoid(x)
% f(x) = 1/(1+exp(-x))
y = 1./(1 + exp(-x));
%y = bsxfun(@rdivide, 1, (1+exp(-x)));

