function out = kmeanstree_search(tree, C, w)
if tree.is_leaf
    out.ids = tree.id_self;
    out.depth = tree.depth;
    return;    
end

dist = compute_mtx_innerproduct(tree.centers, tree.bc_norm2, C, w);
[~, id_child] = max(dist);
% [~, id_child ] = max( dist_metric(tree.centers, tree.bc_norm2, C, w) );
out = kmeanstree_search( tree.subtree{id_child}, C, w );
end
