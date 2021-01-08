clc
clear

num_points = 10000;
num_features = 128;
num_clusters = 50;
num_to_choose = 10;
num_iterations= 10;

all_parameters = (1:10) * 10000;

params.card = num_to_choose;
params.depth = 100;
params.min_items = 200;
params.num_trees = 1;
params.num_cls = 50;
params.mips = 'mips_ktree';

methods = {};
methods(end+1,:) = {'exact', 'ExactGreedy'};
methods(end+1,:) = {'mips_ktree', 'MIPS-ktree'};


% declare the output structure.
results = repmat(...
  struct(...
  'optval', zeros(num_iterations, length(all_parameters)), ...
  'timall', zeros(num_iterations, length(all_parameters)), ...
  'times_preprocess', zeros(1, length(all_parameters)) , ...
  'times_greedy', zeros(num_iterations, length(all_parameters)), ...
  'chosen', zeros(num_iterations, length(all_parameters), params.card), ...
  'st', []),...
  1, length(methods));

base_idx = find(contains(methods(:,1), 'exact'));

for i = 1 : length(all_parameters)
  
  num_points = all_parameters(i);
  fprintf('N = %d\n', num_points);
  for k = 1 : num_iterations
  
    feature_matrix = generate_nonuniform_matrix(...
      num_points, num_features, num_clusters)';
  
    params.depth = fix(log(num_points) * 2);
    params.num_stoch_sample = fix(log(num_points) * 15);
  
    for j = 1 : size(methods, 1)
      if contains(methods{j,1}, 'mips')
        tic
        mips_structure = mips_generate(feature_matrix, params);
        mips_time = toc;
        results(j).times_preprocess(i) = mips_time;
      end
    end
  
    for j = 1 : size(methods, 1)
      if contains(methods{j,1}, 'exact')
        [chosen, tim] = greedy_exact(feature_matrix, params);
      elseif contains(methods{j,1}, 'mips')
        [chosen, tim] = greedy_mips(feature_matrix, params, [], mips_structure);
      end

      results(j).optvals(k,i) = logdet_submtx(feature_matrix, chosen);
      results(j).times_greedy(k,i) = tim.post;
      results(j).chosen(k,i,:) = chosen;
    end
  end
  
  fprintf('[dim = %d] \n', num_points);  
  % print the result of average of iterations
  fprintf("%-12s: " ,'prepro time');
  for j = 1 : length(methods)
    fprintf('%11.6f, ', results(j).times_preprocess(i));
  end
  fprintf("\n%-12s: " ,'greedy time');
  for j = 1 : length(methods)
    fprintf('%11.6f, ', mean(results(j).times_greedy(:,i)));
  end
  fprintf("\n%-12s: ",'log-det');
  for j = 1 : length(methods)
    fprintf('%11.2f, ', mean(results(j).optvals(:,i)));
  end
  fprintf("\n%-12s: ",'opt-ratio');
  for j = 1 : length(methods)
    fprintf('%11.4f, ', mean(results(j).optvals(:,i)./ ...
      results(base_idx).optvals(:,i)));
  end
  fprintf("\n");
  
end
