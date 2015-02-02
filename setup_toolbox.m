function setup_toolbox()
% get toolbox root path
[a, b, c] = fileparts(mfilename('fullpath'));
p_root = a;

% add following directories
add_path( p_root );
add_path( fullfile(p_root, 'toolbox') );

add_path( fullfile(p_root, 'toolbox/DBNLIB') );

add_path( fullfile(p_root, 'toolbox/CDBNLIB') );
add_path( fullfile(p_root, 'toolbox/CDBNLIB/mex') );

add_path( fullfile(p_root, 'toolbox/Softmax') );
add_path( fullfile(p_root, 'toolbox/Softmax/minFunc') );
add_path( fullfile(p_root, 'toolbox/Softmax/minFunc/logistic') );

add_path( fullfile(p_root, 'data') );
add_path( fullfile(p_root, 'data/mnist') );
add_path( fullfile(p_root, 'data/MITcoast') );

add_path( fullfile(p_root, 'model') );



function add_path(p)
fprintf('add path: %s\n', p);
addpath(p);
