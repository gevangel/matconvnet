function [net, info] = cnn_mnist_reg(varargin)
% CNN_MNIST Demonstrates MatConvNet on MNIST

%run(fullfile(fileparts(mfilename('fullpath')),...
%    '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.expDir = fullfile('data','mnist-baseline');
[opts, varargin] = vl_argparse(opts, varargin);

opts.dataDir = 'data/mnist';
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.batchNormalization = false ;
opts.networkType = 'simplenn';
opts.modelType = 'cnn_1_layer';  
opts.train = struct();
opts.numEpochs = 50;
opts.batchSize = 300;

% Regularization
opts.useReg = true;
%if opts.useReg
opts.regType = 'dreg'; %'l2';
opts.regParam = 10;
opts.useWeightNorm = false;
%end

opts = vl_argparse(opts, varargin);

% if ~opts.useReg, opts = rmfield(opts, {'regParam', 'regType'}); end
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath); 
    % set validation set size: first 10K images for MNIST      
    if strfind(opts.imdbPath, '/mnist/')
        imdb.images.set(1:10000) = 2;
    end
else
    if strfind(opts.imdbPath, '/mnist/')
        imdb = getMnistImdb(opts); % MNIST        
    elseif strfind(opts.imdbPath, '/affmnist/')
        imdb = getAffMnistImdb(opts, 0.02); % Affine MNIST
    else        
        imdb = getRotMnistImdb(opts); % Rotated MNIST
    end
    if ~exist(fileparts(opts.imdbPath), 'dir')
        mkdir(fileparts(opts.imdbPath));
    end
    % mkdir(opts.expDir);
    save(opts.imdbPath, '-struct', 'imdb');
end

% --------------------------------------------------------------------
%                                                       Initialize net
% --------------------------------------------------------------------

% additional options for initialization
varargin_init =  {'batchNormalization', opts.batchNormalization, ...
    'networkType', opts.networkType, ...
    'modelType', opts.modelType, ...
    'numEpochs', opts.numEpochs, ...
    'batchSize', opts.batchSize, ...
    'useWeightNorm', opts.useWeightNorm};

net = cnn_mnist_init_mod(varargin_init{:});

% Weight normalization
if opts.useWeightNorm
    net.meta.trainOpts.useWeightNorm = opts.useWeightNorm; % conv weight normalization 
end

% Regularization
if opts.useReg
    net.meta.trainOpts.useReg = opts.useReg; 
    net.meta.trainOpts.regParam = opts.regParam;
    net.meta.trainOpts.regType = opts.regType;
    net.meta.trainOpts.weightDecay = 0; % zero explicit weight decay
    
    regTypeStr = {'dreg-m', 'dreg-mc', 'sreg'};
    if any(strcmp(opts.regType, regTypeStr))
        % Parameters for multiple orbits
        n = length(net.layers);
        % groupSize = 5; % same across layers
        m = 5; % number of orbits: for now same in all layers
        for l=1:n
            if strcmp(net.layers{l}.type, 'conv') && l~=n-1
               nFiltersLayer = size(net.layers{l}.weights{1}, 4);
               
               if mod(nFiltersLayer, m)
                   setm = m*ones(1, nFiltersLayer); 
                   setm(1, mod(nFiltersLayer, m)*m) = kron(1:m-1, ones(1, mod(nFiltersLayer, m)*m));
               else
                   setm = kron(1:m, ones(1, nFiltersLayer/m));
               end                                 
              
               % m = length(setm); 
               net.layers{l}.groups = setm;
               net.layers{l}.groupSize = nFiltersLayer/m;            
                         
            end
            % Add goup info to the maxout layer
             if strcmp(net.layers{l}.type, 'maxout')
                 net.layers{l}.groups = setm;
             end            
        end        
    end
end

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
    'train', find(imdb.images.set == 1),...
    'val', find(imdb.images.set == 2));

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

