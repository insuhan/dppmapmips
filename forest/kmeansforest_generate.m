function forest = kmeansforest_generate(B, params)
forest = cell(1, params.num_trees);
for i = 1 : params.num_trees
    forest{i} = kmeanstree_generate(B, 1:size(B,2), params);
end
end