function[Y, tim, structure] = greedy_stoch(B, params, w, structure)
% This is implementation of Mirzasoleiman, Baharan, et al.
% "Lazier than lazy greedy." AAAI 2015.
% INPUT
%   B : [D x N] features matrix
%   params.card : cardinality of output set
%   params.num_stoch_sample : number of random item selection
% OUTPUT
%   Y : selected subset with cardinality K
%   tim : post-processing time [sec]

if ~isfield(params, 'num_stoch_sample')
  error('Invalid approximation option.');
end

% set up for personalized algorithm
if ~exist('w','var')
  w = [];
end

params.mips = 'mips_random';

t1 = tic;
% preprocess for solving MIPS if no structure is given
if ~exist('structure','var')
  structure = mips_generate(B, params);
end
tim.pre = 0.0;

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