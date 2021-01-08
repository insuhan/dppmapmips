function [idx_cls, C] = kmeans_custom(X, num_cluster)
if num_cluster == -1 % this is for debug
  C = sum(X,2) / size(X,2);
  idx_cls = ones(size(X,2),1);
  return;
end

opts = statset('MaxIter', 1000);
[idx_cls, C] = kmeans(X', num_cluster, 'Options', opts, 'Replicates',1, ...
  'Start','uniform');
C = C';
end
