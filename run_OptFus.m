% Add all the path to the folders
addpath(genpath('.'))
%%
% Load datasets CMOS,FX10,FX17,HC,FEN,and RGB
r=[ 10,20,50,100,200];
H_Res=ImRescale(RGB,1000);
MP_H2=Make_Morphology_profile(H_Res,r,r,'MPr');
clear H_Res
%% Estimating the fused features 
Fus=OptFus(15,[1.5,0.1,0.5,1.5,0.2,1.5],.1,10,40,CMOS,FX10,FX17,HC,FEN,MP_H2);
%% Use a classifier like SVM to classify the features 
%[acc_Mean,acc_std,CM]=SVM_ntimes_overal(Fus,TR_label,TE_label);acc_Mean
%imagesc(CM(:,:,1));axis image;axis off;colormap('jet')
