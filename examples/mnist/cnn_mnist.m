function [net, info] = cnn_mnist(varargin)
% CNN_MNIST Demonstrated MatConvNet on MNIST

run(fullfile(fileparts(mfilename('fullpath')),...
    '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.expDir = fullfile('data','mnist-baseline') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = 'data/mnist' ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.useBatchNorm = false ;
opts.networkType = 'simplenn';
opts.train = struct();
opts.numEpochs = 20;

opts = vl_argparse(opts, varargin);
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    imdb = getMnistImdb(opts) ;
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end

% --------------------------------------------------------------------
%                                                       Initialize net
% --------------------------------------------------------------------

net = cnn_mnist_init('useBatchNorm', opts.useBatchNorm, ...
    'numEpochs', opts.numEpochs, ...
    'networkType', opts.networkType) ;

net.meta.classes.name = imdb.meta.classes;
% arrayfun(@(x)sprintf('%d',x), 1:10, 'UniformOutput', false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

switch opts.networkType
    case 'simplenn', trainfn = @cnn_train ;
    case 'dagnn', trainfn = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
    case 'simplenn'
        fn = @(x,y) getSimpleNNBatch(x,y) ;
    case 'dagnn'
        bopts = struct('numGpus', numel(opts.train.gpus)) ;
        fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if opts.numGpus > 0
    images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;
