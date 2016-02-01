function Y = vl_nnreg(varargin)
%VL_NNREG Adding regularization on the loss function.
%
%   Y = VL_NNREG(net) computes the regularization term for the loss
%   function.
%
%   Y = VL_NNREG(w) computes the weight-dependend gradient contribution on
%   dz/dw.
%
%   VL_NNREG() supports several regularizers, which can be selected
%   by using the option `type` described below.
%
%   l2/Tikhonov:: `l2`
%
%   l1/Sparse:: `l1`
%
%   Single Orbit:: `orb`
%
%   Multiple Orbits:: `morb`

opts.regType = 'l2';
opts.gpus = [];

% morb regularization options
opts.groups = [];
opts.groupSize = [];

opts = vl_argparse(opts, varargin(2:end));
s = 0.01; % orbit reg. parameter

if isstruct(varargin{1}) % opts.compGrad
    
    % Forward pass: Loss term computation
    net = varargin{1};
    n = numel(net.layers);    
    Y = 0;
    
    switch lower(opts.regType)
        case 'l2'
            for l=1:n
                if strcmp(net.layers{l}.type, 'conv')
                    Y = Y + 0.5*sum(sum(sum((net.layers{1}.weights{1}).^2))); % Frobenius squared
                end
            end
        case 'l1'
            % do nothing for now
            % Yr = Yr + sum(sum(sum(abs(net.layers{1}.weights{1})))); % Frobenius squared
            
        case 'orb'
            % Orbit regularizer: only apply on the representation, not the
            % classifier weights
            for l=1:n

                if strcmp(net.layers{l}.type, 'conv') && l~=n-1                    
                    F = net.layers{l}.weights{1};
                    weightSize = [size(F,1) size(F,2) size(F,3) size(F,4)];
                    
                    k = weightSize(4); 
                    
                    % vectorize the weight tensor/cube
                    W = reshape(F(:,:,:,:), [weightSize(1)*weightSize(2)*weightSize(3), k]);
                    Y = Y + regW(W, k, s);
                    %                     W = reshape(F(:,:,:,:), [filterSize(1)*filterSize(2), filterSize(3), k]);
                    %                     % iterate over filter dimensions
                    %                     for d = 1:size(W,2)
                    %                         Y = Y + regW(squeeze(W(:,d,:)), k, s);
                    %                     end
                    
                end
            end
            
          case 'morb'
            % Multiple orbit regularizer
            Y = 0;
            for l=1:n              
                
                if strcmp(net.layers{l}.type, 'conv') && l~=n-1                    
                    F = net.layers{l}.weights{1};
                    nGroups = unique(net.layers{l}.groups); % number of orbits in layer
                    
                    weightSize = [size(F,1) size(F,2) size(F,3) size(F,4)];
                    volWeightTensor = weightSize(1)*weightSize(2)*weightSize(3);
                    
                    % auxiliary matrix
                    k = net.layers{1}.groupSize;
                    if ~isempty(opts.gpus) && opts.gpus >= 1
                        Ik = gpuArray(eye(k));
                        % Ik = gpuArray.speye(k);
                    else
                        Ik = sparse(eye(k)); %Ik = eye(k);
                    end
                    E = kron(Ik, ones(k));
                    
                    % vectorize the weight tensor/cube 
                    W = reshape(F, [volWeightTensor, weightSize(4)]); 
                    
                    % sum over each groups/orbits
                    for g=nGroups
                        Y = Y + regW(W(:, net.layers{l}.groups==g), k, s, E);
                    end
                    % Y = regW_mult(W, net.layers{l}.groups, k, s, E);
                end
            end   
            
        otherwise
            error('Unknown regularizer ''%s''.', opts.regType);
    end
    
else
    % Backward pass: Gradient of regularizer (to be added to dz/dw)
    grad_in = varargin{1};
    
    switch lower(opts.regType)
        case 'l2'
            Y = grad_in;
        case 'l1'
            % do nothing for now
            
        case 'orb'
            % Orbit regularizer
            F = grad_in;
            weightSize = [size(F,1) size(F,2) size(F,3) size(F,4)];
            %if weightSize(1)~=1 && weightSize(2)~=1
            k = weightSize(4);
            
            % create the auxiliary matrices
            % TO-DO: this should be external from this function also!
            [C, R] = gradW_opt_aux(k);
            
            if ~isempty(opts.gpus) && opts.gpus >= 1
                C = gpuArray(C);
                R = gpuArray(R);
                Ik = gpuArray(eye(k));
                % Ik = gpuArray.speye(k);
            else
                Ik = sparse(eye(k)); %Ik = eye(k);
            end
            CRt = R'*C';
            E = kron(Ik, ones(k));
            
            W = reshape(F(:,:,:,:), [weightSize(1)*weightSize(2), weightSize(3), k]);
            
            % iterate over filter dimensions
            for d = 1:weightSize(3)
                vecG = gradW_opt_1(double(squeeze(W(:,d,:))), k, s, Ik, E, CRt);
                Y(:,:,d,:) = reshape(vecG, [weightSize(1), weightSize(2), 1, k]);
            end
            % profile viewer
            
            % W = reshape(F(:,:,:,:), [weightSize(1)*weightSize(2)*weightSize(3), k]);
            % Y = gradW_opt(double(W), k, s);
            
            %else
            %    Y = 0;
            %end
                        
        case 'morb'
            % Multiple orbit regularizer
                        
            F = grad_in;
            
            weightSize = [size(F,1) size(F,2) size(F,3) size(F,4)];
            W = reshape(F(:,:,:,:), [weightSize(1)*weightSize(2), weightSize(3), weightSize(4)]);
            Y = zeros(weightSize(1), weightSize(2), weightSize(3), weightSize(4));            
            nGroups = unique(opts.groups); % number of orbits in layer
            k = opts.groupSize;   
                    
            % create the auxiliary matrices
            % TO-DO: this should be external from this function also!
            [C, R] = gradW_opt_aux(k);
            
            if ~isempty(opts.gpus) && opts.gpus >= 1
                C = gpuArray(C);
                R = gpuArray(R);
                Ik = gpuArray(eye(k));
                % Ik = gpuArray.speye(k);
                Y = gpuArray(Y);
            else
                Ik = sparse(eye(k)); %Ik = eye(k);                
            end
            CRt = R'*C';
            E = kron(Ik, ones(k));            
                       
            % iterate over filter dimensions
            for d = 1:weightSize(3)
                % iterate over groups
                for g=1:nGroups
                    ind_g = opts.groups==g;
                    vecG = gradW_opt_1(double(squeeze(W(:, d, ind_g))), k, s, Ik, E, CRt);                                      
                    Y(:,:,d,ind_g) = reshape(vecG, [weightSize(1), weightSize(2), 1, k]);
                end
            end            
            
        otherwise
            error('Unknown regularizer ''%s''.', opts.regType);
    end
    
end

end
