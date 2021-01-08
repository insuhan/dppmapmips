function B = get_nonuniform_cluster_matrix(n, m, k)
B = zeros(n, m);
num_points_in_clusters = get_num_points_in_clusters(n, k);
mean_scaling = 10/ sqrt(m);
indices = [0 cumsum(num_points_in_clusters)];
for i = 1:k
    idx = indices(i) + 1 : indices(i+1);
    points_per_cluster = length(idx);
    B(idx,:) = mvnrnd(mean_scaling * randn(m, 1), ...
    0.1 * diag(ones(m, 1)), points_per_cluster);    
end
B = B(randperm(n),:);
end

function num_points_in_clusters = get_num_points_in_clusters(n, k)
lam = 10;
num_points_in_clusters = poissrnd(lam, 1, k);

num_points_in_clusters = ...
    round(num_points_in_clusters * (n / sum(num_points_in_clusters)));

diff = n - sum(num_points_in_clusters);
[~, idx] = max(num_points_in_clusters);
num_points_in_clusters(idx) = num_points_in_clusters(idx) + diff;
end