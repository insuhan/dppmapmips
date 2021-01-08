function y = logdet_submtx(B, Y, w)
if exist('w','var')
  B = w.*B;
end
A = B(:,Y)' * B(:,Y);
A = (A + A')/2;
try
  L = chol(A);
catch
  L = chol(A + 1e-5 * eye(size(A)));
end
y = 2*sum(log(diag(L)));
end
