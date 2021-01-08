function out = kmeansforest_search(forest, Q, w)
res = cell(1, length(forest));
for i = 1: length(forest)
    res{i} = kmeanstree_search(forest{i}, Q, w);
end
out.ids = unique(cell2mat(cellfun(@(x) x.ids, res,'UniformOutput',0)));
end