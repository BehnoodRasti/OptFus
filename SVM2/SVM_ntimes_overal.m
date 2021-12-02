function [Indian_cl_acc_Mean,Indian_cl_acc_std,classification_map]=SVM_ntimes_overal(Image,Train,Test)



n=size(Train,3);
for Num_itr=1:n
T=Train(:,:,Num_itr);
T1=Test(:,:,Num_itr);
%  [T1,T]=extract_test_train_of_GrounTruth(output_test_filename,output_train_filename,Ground_truth_im,percent,options);
%  output_test_filename=str2var('output_test_filename')
%  output_test_filename=str2var('output_train_filename')
% load TE
% load TR
%T=train_set_im;
[nx,ny,nz]=size(T);
%train_label=reshape(T,nx*ny,nz)';
%T1=test_set_im;
%test_label=reshape(T1,nx*ny,nz)';
%tic
x1=Image(1:nx,1:ny,:);
[nx,ny,nz]=size(T);
train_label=reshape(T,nx*ny,nz)';


[s1,s2,s3]=size(x1);
Data=reshape(x1,s1*s2,s3)';

 for i=1:s3
     Data(i,:)=double(mat2gray(Data(i,:)));
 end

train_labels=double(train_label(train_label>0));
X_1=Data(:,train_label>0);
%
heart_scale_inst=X_1';
heart_scale_label=train_labels';
%[heart_scale_label, heart_scale_inst] = libsvmread('../heart_scale');
%parameter=sprintf('-c %f -g %f -t 2 -m 500 -q',Ccv,Gcv);
%model = svmtrain(heart_scale_label, heart_scale_inst, '-c 1 -g 0.07');

% learn the SVM parameter
[Ccv Gcv cv cv_t]=cross_validation_svm(heart_scale_label,heart_scale_inst);
parameter=sprintf('-c %f -g %f -t 2 -m 500 -q',Ccv,Gcv);

model = svmtrain(heart_scale_label, heart_scale_inst,parameter);



[nx1,ny1,nz1]=size(T1); 
test_label=reshape(T1,nx1*ny1,nz1)';
test_labels=double(test_label(test_label>0));
X2=Data(:,test_label>0);

x_2=X2';y=test_labels';
heart_scale_label=y;
heart_scale_inst=x_2;
[predict_label, accuracy, dec_values] = svmpredict(heart_scale_label, heart_scale_inst, model); % test the training data
%

%
heart_scale_inst=Data';
heart_scale_label=zeros(s1*s2,1);
[predict_label, accuracy1, dec_values] = svmpredict(heart_scale_label, heart_scale_inst, model); % test the training data
classification_map=reshape(predict_label,s1,s2);

%---------------------------

t =predict_label;

[sortedlabels,sidx]=sort(test_label);

Nc=length(unique(test_labels));

for i=1:Nc
    cl=find(sortedlabels==i);
    s=cl(1);e=cl(length(cl));
    sv=t(sidx);
    pcl=sv(s:e);
    for j=1:Nc
        C(j,i)=length(find(pcl==j));
    end
    Cacc(i)=length(find(pcl==i))/length(pcl)*100;
    clear cl pcl;
end
N=sum(sum(C));
sumC=sum(C);
sumR=sum(C');
S=0;
for i=1:Nc
    acc(i)=C(i,i)/sumC(i)*100;
    S=S+sumC(i)*sumR(i);
end
trace(C);
meanacc=mean(acc);
OA=trace(C)/N*100;
Po=trace(C)/N;
Pe=S/N^2;
kappa=(Po-Pe)/(1-Pe)*100;
%------------------------------
%figure;imagesc(classification_map);axis image;axis off;colormap(vivid(17));caxis([0 16]);%(vivid(max(T(:))));%colorbar('location','southoutside');
% title(['Classification using RF,meanacc=',num2str(meanacc), ',OA=',num2str(OA), ',kappa=',num2str(kappa)])
% figure;
% classif=label2color(classification_map,'uni');
 predict_labels=double(predict_label(test_label>0));
 [oa ua pa K confu]=confusion(test_labels',predict_labels);
IndMat1(:,Num_itr)=[pa;mean(pa);oa;K];
IndMat2(:,Num_itr)=[ua;mean(ua)];
end
Indian_cl_acc_Mean=mean(IndMat1,2);
Indian_cl_acc_std=std(IndMat1,0,2);