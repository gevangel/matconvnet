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
%   Symmetry/Orbits:: `sym`

opts.regType = 'l2';
opts.gpus = [];
opts = vl_argparse(opts, varargin(2:end));

if isstruct(varargin{1}) % opts.compGrad
    % Forward pass: Loss term computation
    net = varargin{1};
    n = numel(net.layers);
    
    Y = 0;
    switch lower(opts.regType)
        case 'l2'
            for i=1:n
                l = net.layers{i};
                if strcmp(l.type, 'conv')
                    Y = Y + 0.5*sum(sum(sum((net.layers{1}.weights{1}).^2))); % Frobenius squared
                end
            end
        case 'l1'
            % do nothing for now
            % Yr = Yr + sum(sum(sum(abs(net.layers{1}.weights{1})))); % Frobenius squared
            
        case 'sym'
            % Orbit regularizer: only apply on the representation, not the
            % classifier weights
            for i=1:n
                l = net.layers{i}; 
                s = 0.01; % reg. parameter
                if strcmp(l.type, 'conv') && i~=n-1                    
                    F = l.weights{1};
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
            
        otherwise
            error('Unknown regularizer ''%s''.', opts.regType);
    end
    
else
    % Backward pass: Gradient of regularizer (to be added to dz/dw)
    l = varargin{1};
    
    switch lower(opts.regType)
        case 'l2'
            Y = l;
        case 'l1'
            % do nothing for now
            
        case 'sym'
            % Orbit regularizer
            F = l;
            weightSize = [size(F,1) size(F,2) size(F,3) size(F,4)];
            s = 0.01; % reg. parameter
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
            
        otherwise
            error('Unknown regularizer ''%s''.', opts.regType);
    end
    
end

end
