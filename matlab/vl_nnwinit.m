function W = vl_nnwinit(sx, sy, sz, nFilters, typeInit, varargin)
% VL_NNWINIT Different schemes for weight initialization of a nn layer
%
% W = vl_nnwinit(sx, sy, L1, L2, typeInit) initializes in W the 
% (sx x sy x L1) weights for L2. This is equivalent to initilizing 
% the weights for (sx x sy) filters each of dimension L1 = sz in a layer with 
% L2 = nFilters.
% 
% typeInit is the type of initialization which can be:
%   'initrandn': random normal (default)
%   'inituni': standard variance heuristic with rand weights
%   'initrand': normalized variance (Glorot, Bengio, 2014)
%   'initrelu': for ReLU units (He, Zhang,...,2015) 
%    ...
%
% W = vl_nnwinit(sx, sy, L1, L2, 'initrandn', f) scales the random weights
% by f, i.e. sets the mean of the normal to 1/f.

if isempty(varargin), varargin{1} = 1/100; end
if nargin<5, typeInit = 'initrandn'; end

switch typeInit
    
    case 'inituni'
        % standard heuristic i.e. w \in U [-1/sqrt(n), 1/sqrt(n)]
        L_in = sx*sy;
        
        epsilon_init = 1/sqrt(L_in);
        W =  - epsilon_init + 2*epsilon_init*rand(sx, sy, sz, nFilters, 'single');
        
    case 'initrand'
        % normalized initialization (Glorot-Bengio 2014)
        L_in = sx*sy; L_out = nFilters;
        
        epsilon_init = sqrt(6)./(sqrt(L_out + L_in));
        W =  - epsilon_init + 2*epsilon_init*rand(sx, sy, sz, nFilters, 'single');
        
    case 'initrelu'
        % intitialization for relu neurons ("Delving Deep into Rectifiers")
        W = randn(sx, sy, sz, nFilters, 'single')./sqrt(2/sx*sy);
        
    case 'initvar'
        W = randn(sx, sy, sz, nFilters, 'single')./sqrt(sx*sy);
        
    case 'initrandperm'
        % random permutation of vector
        r = randn(sx, sy, sz);
        W = single(zeros(sx, sy, sz, nFilters));
        for i=1:nFilters
            ix = randperm(sx);
            iy = randperm(sy);
            iz = randperm(sz);
            W(:,:,:,i) = r(ix, iy, iz);
        end
        
    otherwise % case 'randn'
        % default random weights
        f = varargin{1}; % scalar factor
        W = f*randn(sx, sy, sz, nFilters, 'single');
end

% W1 = squeeze(W(:,:,1,:));
% [size_wx, size_wy, filters_w] = size(W1);
% Wd = reshape(W1, size_wx*size_wy, filters_w);
% regW(Wd, filters_w, 0.01)
