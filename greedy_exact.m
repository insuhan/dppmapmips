function[chosen, tim] = greedy_exact(features, params, query_feature)
    % DESCRIPTION
    %   This is implementation of algorithm proposed by Laming Chen et al.,
    %   "Fast greedy map inference for determinantal point process to improve
    %    recommendation diversity." NeurIPS. 2018.
    % INPUT
    %   B : [D x N] features matrix of items (required)
    %   params
    %         .K : cardinality of output set (required)
    %   W : [D x M] features matrix of users (optional)
    % OUTPUT
    %   Y : selected subset with cardinality K
    %   tim : post-processing time [sec]
    % COPYRIGHT
    %   Author: insu.han@kaist.ac.kr (Insu Han)
    
    t1 = tic;
    % set up for personalized algorithm
    if ~exist('query_feature','var')
      customized_features = features;
    else
      customized_features = query_feature .* features;
    end
    
    tim.pre = 0;
    
    % initialize the output
    chosen = [];
    
    % greedy procedure
    t2 = tic;
    for k = 1 : params.card
      if k == 1
        marginal_gain = sum(customized_features.^2, 1);
        max_index = argmax(marginal_gain);
        chosen(end+1) = max_index;
      else
        ei = customized_features(:,max_index)'*customized_features;
        if k > 2
          ei = ei - C(:,max_index)'*C;
        end
        ei = ei / sqrt(marginal_gain(max_index));
        if k == 2
          C = ei;
        else
          C(end+1,:) = ei;
        end
        marginal_gain = marginal_gain - ei.^2;
        max_index = argmax(marginal_gain);
        chosen(end+1) = max_index;
      end
    end
    tim.all  = toc(t1);
    tim.post = toc(t2);
    end
    