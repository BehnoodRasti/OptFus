% -- SCRIPT TEST CLASSIF TEST PEDRAM -- %
% -----  Process the whole image  ----- %

%clear all, close all,clc
% Load the data
%im = imread('FODPSO14.tif'); % -> here you need to change to the image you want to classify
% load Indian_pines_corrected
% im=indian_pines_corrected;
% im = double(reshape(im,145*145,200));
% load('training_testing_indices_Indian');
% 
% r = 1;% We do not need to repeat the experiment several times
% % Just need one training
% x =  im(it(r,:),:);
% y = double(Yt(r,:))';
% [x M m]=scale(x);
% im = scale(im,M,m);
% 
% % learn the SVM parameter
% [Ccv Gcv cv cv_t]=cross_validation_svm(y,x);
% 
% % Learn the SVM
% parameter=sprintf('-c %f -g %f -t 2 -m 500 -q',Ccv,Gcv);
% model=svmtrain(y,x,parameter);
% 
% % Do the prediction
% res = svmpredict(ones(145*145,1),im,model); % here we do not know the label of all the pixels, so we just provide a vector of ones
% 
% res = reshape(res,145,145); % get the original image size
% 
% classif = label2color(res,'uni');

clear all; close all;clc
% Load the data
%im = imread('FODPSO14.tif'); % -> here you need to change to the image you want to classify
%load Indian_pines_corrected
%im=indian_pines_corrected;
im=imread('IndianPine_BandReducted_16bit.tif');
im = double(reshape(im,145*145,185));
TR = imread('IndianTR123_temp123.tif');

r=1;
it = find(TR~=0);
Yt = im(it);

x =  im(it,:);
y = double(Yt);
[x M m]=scale(x);
im = scale(im,M,m);

% learn the SVM parameter
[Ccv Gcv cv cv_t]=cross_validation_svm(y,x);

% Learn the SVM
parameter=sprintf('-c %f -g %f -t 2 -m 500 -q',Ccv,Gcv);
model=svmtrain(y,x,parameter);

% Do the prediction
res = svmpredict(ones(145*145,1),im,model); % here we do not know the label of all the pixels, so we just provide a vector of ones

res = reshape(res,145,145); % get the original image size

%classif = label2color(res,'uni');