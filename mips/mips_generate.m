function structure = mips_generate(B, params)
switch(params.mips)
  case 'mips_exact'
    mips = [];
  case 'mips_random'
    mips = [];
  case 'mips_kmeans'
    if ~isfield(params,'num_clusters')
      error('Require num_clusters parameter');
    end
    % perform kmeans clustering
    [cluster_index, B_centers] = kmeans(B, params.num_clusters);
    
    % compute the number of items in each cluster
    cluster_sizes = histc(cluster_index, unique(cluster_index));
    
    % find indices of items in each cluster
    [~, cluster_id_to_point_id] = sort(cluster_index);
    cluster_point_id = mat2cell(cluster_id_to_point_id', 1, cluster_sizes);
    
    % store the information to structure
    mips.indices  = cluster_index;
    mips.sizes    = cluster_sizes;
    mips.point_id = cluster_point_id;
    mips.centers  = B_centers;
    mips.bc_norm2 = sum(B_centers.^2, 1);
    
    assert(size(B_centers,1)==size(B,1));
  case 'mips_ktree'
    if ~isfield(params,{'num_cls','depth','num_trees', 'min_items'})
      error('Not enough parameters');
    end
    
    % create the kmeans forest structure (see forest/kmeansforest_generate.m)
    mips = kmeansforest_generate(B, params);
    
  otherwise
    error('Choose a valid option.');
end

structure.mips    = mips;
structure.b_norm2 = sum(B.^2, 1);
end