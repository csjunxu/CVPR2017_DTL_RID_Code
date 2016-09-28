function [P, patch_size, nch] = sample_patches(im, patch_num, Patch_Channel, R_thresh)
if Patch_Channel == 3,
    patch_size = 6;
    disp('RGB image sampled!');
elseif Patch_Channel ==1
    im = rgb2ycbcr(im);
    im = im(:,:,1);
    patch_size = 8;
    disp('Grayscale image sampled!');
end

[nrow, ncol, nch] = size(im);

x = randperm(nrow-2*patch_size-1) + patch_size;
y = randperm(ncol-2*patch_size-1) + patch_size;

[X,Y] = meshgrid(x,y);

xrow = X(:);
ycol = Y(:);

im = double(im);

P = [];

idx=1;ii=1;n=length(xrow);
while (idx < patch_num) && (ii<=n),
    row = xrow(ii);
    col = ycol(ii);
    
    patch = im(row:row+patch_size-1,col:col+patch_size-1,:);
    
    % check if it is a stochastic patch
    np = patch(:) - mean(patch(:));
    npnorm=sqrt(sum(np.^2));
    np_normalised=reshape(np/npnorm,patch_size,patch_size, nch);
    % eliminate that small variance patch
    if var(np)>0.001
        % eliminate stochastic patch
        if dominant_measure(np_normalised)>R_thresh
            %if dominant_measure_G(Lpatch1,Lpatch2)>R_thresh
            P = [P np];
            idx=idx+1;
        end
    end
    ii=ii+1;
end

fprintf('sampled %d patches.\r\n',patch_num);
end

function R = dominant_measure(p)
% calculate the dominant measure
% ref paper: Eigenvalues and condition numbers of random matries, 1988
% p = size n x n patch

hf1 = [-1,0,1];
vf1 = [-1,0,1]';
% Gx = conv2(p, hf1,'same'); %original
% Gy = conv2(p, vf1,'same'); 
Gx = convn(p, hf1,'same'); % changed by JX 20160926
Gy = convn(p, vf1,'same');

G=[Gx(:),Gy(:)];
[U, S, V]=svd(G);

R=(S(1,1)-S(2,2))/(S(1,1)+S(2,2));

end
