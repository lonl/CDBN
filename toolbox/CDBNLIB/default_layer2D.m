function layer = default_layer2D()
%-------------------------------------------------------------------------------%
%                        Default Parameters Setting                             %
%-------------------------------------------------------------------------------%

% LAYER IS A STRUCT
layer =  struct;             

% DEFAULT PARAMETERS SETTING
layer.n_map_v       = 1;        % NUMBER OF VISIBLE FEATURE MAPS 
layer.n_map_h       = 9;        % NUMBER OF HIDDEN FEATURE MAPS
layer.s_filter      = [7 7];    % SIZE OF FILTER
layer.s_pool        = [2 2];    % SIZE OF POOLING
layer.n_epoch       = 20;       % NUMBER OF ITERATION
layer.learning_rate = 0.05;     % RATE OF LEARNING
layer.sparsity      = 0.02;     % HIDDEN UNIT SPARSITY
layer.lambda1       = 5;        % GAIN OF THE LEARNING RATE FOR WEIGHT CONSTRAINTS
layer.lambda2       = 0.05;     % WEIGHT PENALTY
layer.start_gau     = 0.2;      % GAUSSIAN START
layer.stop_gau      = 0.1;      % GAUSSIAN END
layer.batchsize     = 1;        % SIZE OF BATCH IN TRAINING STEP
layer.n_cd          = 1;        % NUMBER OF GIBBS SAMPLES
layer.momentum      = 0.9;      % GRADIENT MOMENTUM FOR WEIGHT UPDATES
layer.whiten        = 1;        % WHETHER TO BE WHITEN
layer.type_input    = 'Binary'; % INPUT STYPE

layer.matlab_use    = 0;        %-----------------------------------------------%
layer.mex_use       = 1;        % JUST ONE STYPE: MEX OR MATLAB OR CUDA USED HERE
layer.cuda_use      = 0;        %-----------------------------------------------%

                                %-----------------------------------------------% 
layer.stride        = [1 1];    % STRIDE OF FILTER MOVE
                                % YOU SHOULD OBEY THE RULE:
                                % mod = ((size(inputdata)-s_filter),stride) == 0!
                                % AND IN FACT, THE STRIDE PARAMETER ONLY
                                % HAS EFFECTS IN FEEDDORWARD STEP
                                %-----------------------------------------------%
                                
end