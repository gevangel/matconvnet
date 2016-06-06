function info = cnn_mnist_evaluate(varargin)
% taken from CNN_IMAGENET_EVALUATE

%run(fullfile(fileparts(mfilename('fullpath')), ...
%  '..', '..', 'matlab', 'vl_setupnn.m')) ;

% opts.dataDir = '/media/gevang/Data/data/MNIST/idx' ;
% opts.expDir = fullfile('/media/gevang/Data/work/code/cbcl/matconvnet/', 'data/test-mnist-simplenn');
opts.expDir = fullfile('data','mnist');
[opts, varargin] = vl_argparse(opts, varargin);

opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.modelPath = fullfile(opts.expDir, 'mnist-cnn.mat');

opts.networkType = [];
% opts.numFetchThreads = 12 ;

opts.train.batchSize = 100;
opts.train.numEpochs = 1;
opts.train.gpus = [];
opts.train.continue = false;
opts.train.prefetch = true;
opts.train.expDir = opts.expDir;
opts.train.plotStatistics = false; 

opts = vl_argparse(opts, varargin) ;
display(opts);

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
    % else
    %   imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
    %   mkdir(opts.expDir) ;
    %   save(opts.imdbPath, '-struct', 'imdb') ;
end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

net = load(opts.modelPath) ;
if isfield(net, 'net') ;
    net = net.net ;
end

opts.networkType = 'simplenn' ;
net = vl_simplenn_tidy(net) ;
trainfn = @cnn_train ;
net.layers{end}.type = 'softmaxloss' ; % softmax -> softmaxloss

% Run evaluation
% net.meta.trainOpts.numEpochs = net.meta.trainOpts.numEpochs + 1;

[~, info] = trainfn(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    opts.train, ...
    'train', NaN, ...
    'val', find(imdb.images.set==3)) ;


% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
    case 'simplenn'
        fn = @(x,y) getSimpleNNBatch(x,y) ;
        %     case 'dagnn'
        %         bopts = struct('numGpus', numel(opts.train.gpus)) ;
        %         fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% % --------------------------------------------------------------------
% function inputs = getDagNNBatch(opts, imdb, batch)
% % --------------------------------------------------------------------
% images = imdb.images.data(:,:,:,batch) ;
% labels = imdb.images.labels(1,batch) ;
% if opts.numGpus > 0
%     images = gpuArray(images) ;
% end
% inputs = {'input', images, 'label', labels} ;

