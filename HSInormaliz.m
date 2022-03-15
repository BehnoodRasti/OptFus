function normalizedM  = HSInormaliz( X )
[~, ~, numBands] = size(X);
normalizedM=X;
for i=1:numBands
    normalizedM(:,:,i)  = Normalization( X(:,:,i) );
end
end
%%
function X_N=Normalization(X)
[nx1, ny1] = size(X);
RH=reshape(X,nx1*ny1,1);
m = min(X(:));
M = max(X(:));
NRH=(RH-m)/(M-m);
X_N=reshape(NRH,nx1,ny1,1);
end