function [y, inds_bin] = vl_nnmaxout(x, varargin)
% VL_NNMAXOUT CNN maxout layer
%   Y = VL_NNMAXOUT(X) applies maxout to the data X, taking the maximum over
%   all channels in X.
%
%   [Y, INDS] = VL_NNMAXOUT(X, 'groupSize', GROUPSIZE) splits X channels
%   into groups of groupSize and returns the maximum response in each group.
%   INDS is a binary MASK (of the same dimension as X) determining which elements of
%   the DZ/DY will be used in DZ/DX.
%
%   [Y, INDS] = VL_NNMAXOUT(X, 'groups', GROUPS) where GROUPS a vector of
%   indices (indicator) returns the maximum response for each group
%   specified in GROUPS.
%
%   DZDX = VL_NNMAXOUT(X, INDS, DZDY) computes the derivatives DZDX of the
%   block (i.e. of the output relative to input X) projected
%   onto DZDY (of output relative to block output Y), given the indices
%   corresponding to maximum elements from the forward step. DZDX and DZDY
%   have the same dimensions as X and Y respectivey.

opts.groups = [];
opts.groupSize = [];
% [opts, varargin] = vl_argparse(opts, varargin);

backMode = numel(varargin) > 0 && ~ischar(varargin{1}) ;
if backMode
    inds = varargin{1};
    dzdy = varargin{2};
    %  opts = vl_argparse(opts, varargin(2:end));
else
    opts = vl_argparse(opts, varargin);
end

if backMode && isempty(inds)
    warning('vl_nnmaxout: when using in backward mode, the mask or vector of indices should be specified') ;
end

[sa, sb, sc, sd] = size(x);

if ~backMode
    if isempty(opts.groups)
        %% Standard Maxout: group indicator function on the fly
        nGroups = sc/opts.groupSize;
        group_id = 1:nGroups; % unique(opts.groups);
        opts.groups = kron(group_id, ones(1, opts.groupSize));
    else
        %% Modified maxout/groupout
        group_id = unique(opts.groups);
        nGroups = length(group_id);
    end
    
    y = zeros([sa sb nGroups sd]);
    % inds_bin = zeros(sa, sb, sc, sd);
    
    if isa(x,'gpuArray')
        y = single(gpuArray(y));
        % inds_bin = gpuArray(inds_bin);
    end
    
    % For each group -> do pooling
    for g=group_id
        [y(:,:,g,:), max_inds] = max(x(:, :, opts.groups==g, :), [], 3);
        inds(:,:,g,:) = max_inds + nGroups*(g - 1); % vector of indices
    end
    
    % Map tensor map of indexes to tensor map of binary indexes
    inds_bin = inds2mask(inds, group_id, sa, sb, sd, sc);
    if isa(x, 'gpuArray')
        inds_bin = gpuArray(inds_bin);
    end
    
else
    % Back-propagation mode
    y = gradmask(inds, dzdy);
    
    %% debug
    %     a1 = 8; a2 = 2; a3 = 129;
    %     disp(squeeze(inds(a1, a2, :, a3))')
    %     disp(squeeze(dzdy(a1, a2, :, a3))')
    %     disp(squeeze(y(a1, a2, :, a3))')
    %
end

function mask = inds2mask(inds, group_id, sa, sb, sd, sc)
% Converts an M x N x I x B tensor of integer channels I (in [1, K]) to a
% M x N x K x B 'binary' mask for applying gradient bp.

mask = 0;
for g=group_id
    
    a = permute(inds(:,:,g,:), [1, 2, 4, 3]);
    b = gather(a(:));
    n = sa*sb*sd; %length(b);
    M = sparse(1:n, b, 1, n, sc);
    mask = mask + M;
end
mask = reshape(full(mask), [sa, sb, sd, sc]);
mask = single(permute(mask, [1, 2, 4, 3]));
% inds_bin(max_inds + nGroups*(g - 1)) = 1; % binary indicator


function Z = gradmask(inds_bin, dzdy)
% Gradient back-projection: Assigns M x N x I x B values to a (sparse)
% matrix M x N x K x B using the binary matrix inds_bin of size M x N x K x B.

% bring the 'group' dimension first in the 4D arrays
p_inds_bin = permute(inds_bin, [3,1,2,4]);
p_dzdy = permute(dzdy, [3,1,2,4]);

% ind_plain = find(p_inds_bin == 1);
Z = zeros(size(p_inds_bin));
if isa(inds_bin,'gpuArray')
    Z = gpuArray(Z);
end
Z(p_inds_bin == 1) = p_dzdy;
Z = single(permute(Z, [2,3,1,4]));

% % tests...
% a1 = 1; a2 = 2; a3 = 129;
% squeeze(inds(a1, a2, :, a3))'
% squeeze(inds_bin(a1, a2, :, a3))'
% squeeze(dzdy(a1, a2, :, a3))'
% squeeze(Z(a1, a2, :, a3))'


