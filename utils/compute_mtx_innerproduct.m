function dist = compute_mtx_innerproduct( B, b_norm2, C, w)
% dist(i) = bi'*bi - bi'*(C*C')*bi; where bi = w .* B(:,i).

if isempty(w)
  Bhat = B;
  bhat_norm2 = b_norm2;
else
  Bhat = w .* B;
  bhat_norm2 = sum(Bhat.^2, 1);
end

if isempty(C)
  dist = bhat_norm2;
else
  dist = bhat_norm2 - sum((C'*Bhat).^2, 1); 
end
end