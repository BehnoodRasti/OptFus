function Fus=OptFus(r,lambda,lambda_tv,mu,nitr,varargin)
% OptFus -- OptFus: Optical Sensor Fusion For The Classification of Multi-source Data
% OptFus solves
% argmin_X .5*Sum^n_{i=1}||M_i-FV_i'||_F^2+\lambda_n+1||F||_tv 
% 
%
%
% Citation:
% OptFus: Optical Sensor Fusion For The Classification of Multi-source Data: Application to Mineralogical Mapping
% doi: 10.1109/LGRS.2021.3132701
%
%  Inputs
%    r :   Number of fused features
%    lambda: A vector containg the lambda_i's which defines the
%    contribution of each sensor
%    lambda_tv: Total variation regularization parameter
%    mu: Penalty parameter 
%    nitr: Number of iterations
%    varargin: The datasets as 3D matrices (See the demo)
%  Outputs
%    F : Fused futures
%
%
% (c) 2017 Behnood Rasti
% behnood.rasti@gmail.com
lambda=lambda(:);
[n1,m1,~]=size(varargin{1});
for i=1:length(varargin)
    varargin{i}=HSInormaliz(varargin{i});
    varargin{i}=reshape(varargin{i},n1*m1,size(varargin{i},3));
    [~,~,V{i}]=svd(varargin{i},'econ');
end
B=zeros(n1*m1,r);S=B;
for i=1:nitr
    sum1=0;
    for ii=1:length(lambda), sum1=sum1+lambda(ii,1)*varargin{ii}*V{ii}(:,1:r); end
    F=(sum1+mu*(S-B))/(sum(lambda)+mu);
    Temp=reshape(B+F,n1,m1,r);
    Temp1=Temp;
    parfor j = 1:r
        Temp1(:,:,j)=splitBregmanROF(Temp(:,:,j),mu/lambda_tv,.1);
    end
    S=reshape(Temp1,n1*m1,r);
    for ii=1:length(lambda),[A1,~,B1]=svd(varargin{ii}'*F,'econ');V{ii}=A1*B1';end
    B=B+F-S;
    fprintf('%d\n',i)
end
Fus=reshape(F,n1,m1,r);
