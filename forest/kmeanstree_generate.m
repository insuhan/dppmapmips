function tree = kmeanstree_generate(B, id_abs, params, depth)
if nargin < 4
    depth = 1;
end

% construct the leaf node
if depth >= params.depth || ...
        length(id_abs) <= max(params.min_items, params.num_cls)
    tree = struct();
    tree.num_cls = 0;
    tree.centers = [];
    tree.id_self = id_abs;
    tree.id_child = [];
    tree.is_leaf = true;
    tree.num_items = length(id_abs);
    tree.depth = depth;
    return;
end

% kmeans clustering the items
[idx_cls, BC] = kmeans_custom(B, params.num_cls);

% split indices of clustered items
[~, id_loc_rel] = sort(idx_cls);
id_hist = histc(idx_cls, unique(idx_cls));
id_child_rel = mat2cell(id_loc_rel', 1, id_hist);
id_child_abs = cellfun(@(x) id_abs(x), id_child_rel,'UniformOutput',false);

% construct the node
tree = struct();
tree.num_cls = params.num_cls;
tree.centers = BC;
tree.id_self = id_abs;
tree.id_child = id_child_abs;
tree.is_leaf = false;
tree.num_items = length(id_abs);
tree.depth = depth;
tree.bc_norm2 = sum(BC.^2, 1);

% construct children of the node
num_items = 0;
subtree = cell(1, params.num_cls);
for i = 1 : params.num_cls
    subtree{i} = kmeanstree_generate(...
        B(:,id_child_rel{i}),id_child_abs{i}, params, depth+1);
    num_items = num_items + subtree{i}.num_items;
end
tree.subtree = subtree;
tree.num_items = num_items;

end