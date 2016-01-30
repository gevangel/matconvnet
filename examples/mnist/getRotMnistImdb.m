% Prepare the rotated MNIST imdb structure, returns image data with mean image subtracted
%
% See also: getMnistImdb.m

function imdb = getRotMnistImdb(opts)

files = {'mnist_all_rotation_normalized_float_train_valid.amat', ...
    'mnist_all_rotation_normalized_float_test.amat'} ;

if ~exist(opts.dataDir, 'dir')
    mkdir(opts.dataDir);
end

if ~exist(fullfile(opts.dataDir, files{1}), 'file')
    url = sprintf('http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip') ;
    fprintf('downloading %s\n', url);
    gunzip(url, opts.dataDir);
end

%train set
x1 = load(fullfile(opts.dataDir, files{1}));
y1 = x1(:, end) + 1;
x1 = x1(:, 1:end-1);

% test set
x2 = load(fullfile(opts.dataDir, files{2}));
y2 = x2(:, end) + 1;
x2 = x2(:, 1:end-1);

% Remove mean estimated from train set
set = [ones(1, numel(y1)) 3*ones(1, numel(y2))];
data = single(reshape(cat(1, x1, x2)', 28, 28, 1, []));
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean);

imdb.images.data = data;
imdb.images.data_mean = dataMean;
imdb.images.labels = cat(1, y1, y2)';
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'};
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false);