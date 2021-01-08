function[Y, tim, structure] = greedy_mips(B, params, w, structure)

% set up for personalized algorithm
if ~exist('w','var')
  w = [];
end

t1 = tic;
% preprocess for solving MIPS if no structure is given
if ~exist('structure','var')
  tt = tic;
  structure = mips_generate(B, params);
  fprintf('%s generated... [%.4f] sec!\n', params.mips, toc(tt));
end
tim.pre = toc(t1);



% initialize the output
Y = [];
C = [];

t2 = tic;
for k = 1 : params.card
  % search the item and add
  a = mips_search(structure, w, C, B, params);
  Y(end+1) = a;
  
  % update the query matrix
  ba = B(:,a);
  if ~isempty(w)
    ba = w .* ba;
  end
  
  if k == 1
    C = (1/sqrt(sum(ba.^2))) * ba;
  else
    tmp = ba - C*(C'*ba);
    C(:,end+1) = (1/sqrt(ba'*tmp)) * tmp;
  end
end
tim.all  = toc(t1);
tim.post = toc(t2);
end
