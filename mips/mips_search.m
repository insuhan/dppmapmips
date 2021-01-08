function id_argmax = mips_search(structure, w, C, B, params)
    mips    = structure.mips;
    b_norm2 = structure.b_norm2;
    
    switch(params.mips)
      case 'mips_exact'
        % find the exact near neighbor
        dist = compute_mtx_innerproduct(B, b_norm2, C, w);
        id_argmax = argmax( dist );
      case 'mips_random'
        % sample random params.num_stoch_sample indices
    %     rnd_idx = randsample( size(B,2), params.num_stoch_sample );
        rnd_idx = randi(size(B, 2), 1, params.num_stoch_sample);
        rnd_idx = unique(rnd_idx);
        
        % compute the inner-products of selected indices
        dist = compute_mtx_innerproduct( B(:,rnd_idx), b_norm2(:,rnd_idx), C, w);
        id_argmax_in_rnd = argmax( dist );
        id_argmax = rnd_idx( id_argmax_in_rnd );
      case 'mips_kmeans'
        % find the nearest cluster
        distc = compute_mtx_innerproduct( mips.centers, mips.bc_norm2, C, w);
        cluster_id = argmax( distc );
        
        % collect ids of corresponding cluster
        point_ids = mips.point_id{cluster_id};
        
        dist = compute_mtx_innerproduct( B(:,point_ids), b_norm2(:,point_ids), C, w);
        id_argmax_in_cluster = argmax( dist );
        id_argmax = point_ids(id_argmax_in_cluster);
      case 'mips_ktree'
        % find the nearest cluster
        res = kmeansforest_search(mips, C, w);
        
        % collect ids whose cluster id is nearest to the query C
        point_ids = res.ids;
        
        % compute the marginal gain of few samples
        dist = compute_mtx_innerproduct( B(:,point_ids), b_norm2(:,point_ids), C, w);
        id_argmax_in_cluster = argmax( dist );
        id_argmax = point_ids(id_argmax_in_cluster);
      case 'mips_ktreesq'
        res = kmeanstree_square_search(mips, C, w);
        point_ids = res.ids;
        
        dist = compute_mtx_innerproduct( B(:,point_ids), b_norm2(:,point_ids), C, w);
        id_argmax_in_cluster = argmax( dist );
        id_argmax = point_ids(id_argmax_in_cluster);
      otherwise
        error('Choose a valid option.');
    end
    
    end
    
    
    