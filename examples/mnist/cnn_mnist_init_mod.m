function net = cnn_mnist_init_mod(varargin)
% CNN_MNIST_INIT_MOD Initialize various small NNs for MNIST

% Modified from original matconvnet distribution: cnn_mnist_init.m
% TO-DO: merge/replace main cnn_mnist_init.m

opts.useBatchNorm = true;
opts.networkType = 'simplenn';
opts.modelType = 'simple'; % 'lenet_base';

% opts.train = struct();
opts.learningRate = 0.001;
opts.numEpochs = 20;
opts.batchSize = 100;

opts = vl_argparse(opts, varargin);

% rng('default');
% rng(0);
rng('shuffle')

f = 1/100;
net.layers = {};

% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(5, 5, 1, 20, 'single'), zeros(1, 20, 'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'pool', ...
%                            'method', 'max', ...
%                            'pool', [2 2], ...
%                            'stride', 2, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(12, 12, 20, 10, 'single'), zeros(1, 10, 'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'loss') ;


switch opts.modelType
    
    case 'cnn_mnist'
        % matconvnet default for MNIST examples
        
        net.layers{end+1} = struct('type', 'conv', ...
            'weights', {{f*randn(5,5,1,20, 'single'), zeros(1, 20, 'single')}}, ...
            'stride', 1, ...
            'pad', 0) ;
        net.layers{end+1} = struct('type', 'pool', ...
            'method', 'max', ...
            'pool', [2 2], ...
            'stride', 2, ...
            'pad', 0) ;
        net.layers{end+1} = struct('type', 'conv', ...
            'weights', {{f*randn(5,5,20,50, 'single'),zeros(1,50,'single')}}, ...
            'stride', 1, ...
            'pad', 0) ;
        net.layers{end+1} = struct('type', 'pool', ...
            'method', 'max', ...
            'pool', [2 2], ...
            'stride', 2, ...
            'pad', 0) ;
        net.layers{end+1} = struct('type', 'conv', ...
            'weights', {{f*randn(4,4,50,500, 'single'),  zeros(1,500,'single')}}, ...
            'stride', 1, ...
            'pad', 0) ;
        net.layers{end+1} = struct('type', 'relu') ;
        net.layers{end+1} = struct('type', 'conv', ...
            'weights', {{f*randn(1,1,500,10, 'single'), zeros(1,10,'single')}}, ...
            'stride', 1, ...
            'pad', 0) ;
        net.layers{end+1} = struct('type', 'loss') ;
        
        
    case 'cnn_2_layer'
        % simple net used in regularization development exp
        
        net.layers{end+1} = struct('type', 'conv', ...
            'weights', {{f*randn(5, 5, 1, 20, 'single'), zeros(1, 20, 'single')}}, ...
            'stride', 1, ...
            'pad', 0) ;
        net.layers{end+1} = struct('type', 'pool', ...
            'method', 'max', ...
            'pool', [2 2], ...
            'stride', 2, ...
            'pad', 0) ;
        net.layers{end+1} = struct('type', 'conv', ...
            'weights', {{f*randn(12, 12, 20, 20, 'single'), zeros(1, 20, 'single')}}, ...
            'stride', 1, ...
            'pad', 0) ;
        net.layers{end+1} = struct('type', 'relu') ;
        net.layers{end+1} = struct('type', 'conv', ...
            'weights', {{f*randn(1, 1, 20, 10, 'single'), zeros(1, 10,'single')}}, ...
            'stride', 1, ...
            'pad', 0) ;
        net.layers{end+1} = struct('type', 'loss') ;
        
        
    case 'cnn_1_layer'
        % basic 1 layer CNN
        
        nFilters = 25; %5
        net.layers{end+1} = struct('type', 'conv', ...
            'weights', {{f*randn(5, 5, 1, nFilters, 'single'), zeros(1, nFilters, 'single')}}, ...
            'stride', 1, ...
            'pad', 0) ;
        net.layers{end+1} = struct('type', 'pool', ...
            'method', 'max', ...
            'pool', [2 2], ...
            'stride', 2, ...
            'pad', 0) ;
        net.layers{end+1} = struct('type', 'conv', ...
            'weights', {{f*randn(12, 12, nFilters, 10, 'single'), zeros(1, 10,'single')}}, ...
            'stride', 1, ...
            'pad', 0) ;
        net.layers{end+1} = struct('type', 'loss') ;
        
    case 'dnn_1_layer'
        % 1 hidden layer/perceptron
        
        nFilters = 25; %5
        net.layers{end+1} = struct('type', 'conv', ...
            'weights', {{f*randn(28, 28, 1, nFilters, 'single'), zeros(1, nFilters, 'single')}}, ...
            'stride', 1, ...
            'pad', 0) ;
        net.layers{end+1} = struct('type', 'conv', ...
            'weights', {{f*randn(1, 1, nFilters, 10, 'single'), zeros(1, 10,'single')}}, ...
            'stride', 1, ...
            'pad', 0) ;
        net.layers{end+1} = struct('type', 'loss');
end


% optionally switch to batch normalization
if opts.useBatchNorm
    net = insertBnorm(net, 1) ;
    net = insertBnorm(net, 4) ;
    net = insertBnorm(net, 7) ;
end

% Meta parameters
net.meta.inputSize = [28 28 1] ;
net.meta.trainOpts.learningRate = opts.learningRate;
net.meta.trainOpts.numEpochs = opts.numEpochs;
net.meta.trainOpts.batchSize = opts.batchSize;

% Fill in defaul values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
    case 'simplenn'
        % done
    case 'dagnn'
        net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
        net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
            {'prediction','label'}, 'error') ;
    otherwise
        assert(false) ;
end

% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
    'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
