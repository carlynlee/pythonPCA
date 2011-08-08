close all;
%clear all
% Identify columns of data...

%addpath('C:\Documents and Settings\charleshlee\Desktop\SOM\somtoolbox');
tic
nIndex=[116:191]; nni=length(nIndex);
primaryTIndexAll=[4,9:112]; npta=length(primaryTIndexAll);


%load data and define indices
xraw = load('imputed-liver.txt');
xraw=xraw(:,[primaryTIndexAll nIndex]);
for i=1:size(xraw,1)
    xraw(i,:)=xraw(i,:)-mean(xraw(i,:));
end

class_num=2;
primaryTIndexAll=1:npta;
nIndex=npta+(1:nni);
Ngenes=size(xraw,1);
Ns=size(xraw,2);


%create train, validation, test sets from tumor samples
N_class1=size(primaryTIndexAll,2);
[TrainIndclass1,ValIndclass1,TestIndclass1] = dividerand([1:N_class1],0.075,0.33,0.60);
N_class1Train=size(TrainIndclass1,2);
N_class1Val=size(ValIndclass1,2);
N_class1Test=size(TestIndclass1,2);


%create train, validation, test sets from normal samples
N_class2=size(nIndex,2);
[TrainIndclass2,ValIndclass2,TestIndclass2] = dividerand([N_class1+1:Ns],0.075,0.33,0.60);
N_class2Train=size(TrainIndclass2,2);
N_class2Val=size(ValIndclass2,2);
N_class2Test=size(TestIndclass2,2);


%STEP 1) Feature Reduction - pca projection over the tumor samples (class 1)
[Cls1phi Cls1projMatA]=Extr_PCA_Features2(xraw,TrainIndclass1);

%STEP 2) Feature Reduction - pca projection over the normal samples (class 2)
[Cls2phi Cls2projMatA]=Extr_PCA_Features2(xraw,TrainIndclass2);

Nfea=2; %project onto cls1 and cls2 primaries
LiverData=[Cls1projMatA(:,1) Cls2projMatA(:,1)]';
LiverDataNorm = [mapminmax(LiverData(1,:)); mapminmax(LiverData(2,:))]; LiverDataNorm = (LiverDataNorm +1 )./2;
LiverDataNormTargets=[ones(1,N_class1) ones(1,N_class2)*-1; ones(1,N_class1)*-1 ones(1,N_class2)];  LiverDataNormTargets = (LiverDataNormTargets +1 )./2;
targets(1,1:N_class1,:)=1; %construct targets
targets(1,1+N_class1:N_class1+N_class2,:)=0;

%Train Set
TrainSet=LiverDataNorm(:,[TrainIndclass1 TrainIndclass2]);
TrainTargets=targets(:,[TrainIndclass1 TrainIndclass2]);
%label train targets
for k=1:length(TrainSet);GeneNamesTrain{k}=num2str(k);end
for k=1:N_class1Train;SampleLabelsTrain{k,1}='1';end
for k=1:N_class2Train;SampleLabelsTrain{N_class1Train+k,1}='2';end

%Validation Set
ValSet=LiverDataNorm(:,[ValIndclass1 ValIndclass2]);
ValTargets = targets(:,[ValIndclass1 ValIndclass2]);
%label validation targets
for k=1:length(ValSet);GeneNamesVal{k}=num2str(k);end
for k=1:N_class1Val;SampleLabelsVal{k,1}='1';end
for k=1:N_class2Val;SampleLabelsVal{N_class1Val+k,1}='2';end

%Test Set
TestSet=LiverDataNorm(:,[TestIndclass1 TestIndclass2]);
TestTargets = targets(:,[TestIndclass1 TestIndclass2]);
%label test targets
for k=1:length(TestSet);GeneNamesTest{k}=num2str(k);end
for k=1:N_class1Test;SampleLabelsTest{k,1}='1';end
for k=1:N_class2Test;SampleLabelsTest{N_class1Test+k,1}='2';end



num=1
%perfcurve for POD over tumors
[X,Y,T,AUC] =perfcurve(SampleLabelsVal,LiverDataNorm(1,[ValIndclass1 ValIndclass2]), '1');
fpr_POD_tumor=X; tpr_POD_tumor=Y; thresholds_POD_tumor=T; AUC_POD_tumor=AUC;

%perfcurve for POD over normals
[X,Y,T,AUC] =perfcurve(SampleLabelsVal,LiverDataNorm(2,[ValIndclass1 ValIndclass2]), '2');
fpr_POD_normal=X; tpr_POD_normal=Y; thresholds_POD_normal=T; AUC_POD_normal=AUC;

%find index of largest product sens*spc for tumor projections
[max,cut_idxT]=max(   [tpr_POD_tumor.*(1-fpr_POD_tumor)]   ); clear max;
%get threshold, tpr, fpr for selected index
best_cutT=thresholds_POD_tumor(cut_idxT); best_cutT_fpr=fpr_POD_tumor(cut_idxT); best_cutT_tpr=tpr_POD_tumor(cut_idxT);

%find index of largest product sens*spc for normal projections
[max,cut_idxN]=max( [ tpr_POD_normal.*(1-fpr_POD_normal)]); clear max;
%get threshold, tpr, fpr for selected index
best_cutN=thresholds_POD_normal(cut_idxN); best_cutN_fpr=fpr_POD_normal(cut_idxN); best_cutN_tpr=tpr_POD_normal(cut_idxN);


figure(99);hold on;
xlabel('False Positive Rate','fontsize',14); ylabel('True Positive Rate','fontsize',14);
title(['POD ROC; Tumor Train =' num2str(N_class1Train) '; Normal Train =' num2str(N_class2Train) , ...
    '; Tumor Validation =' num2str(N_class1Val) '; Normal Validation =' num2str(N_class2Val) ],'fontsize',14)
str_Tumor=['Tumor (AUC=' num2str(AUC_POD_tumor) ');fpr=' num2str(best_cutT_fpr) ';tpr=' num2str(best_cutT_tpr) ';cutoff=' num2str(best_cutT) ];
str_Normal=['Normal (AUC=' num2str(AUC_POD_normal) ');fpr=' num2str(best_cutN_fpr) ';tpr=' num2str(best_cutN_tpr) ';cutoff=' num2str(best_cutN) ];

plot(fpr_POD_tumor,tpr_POD_tumor,'b','LineWidth',2); 
plot(fpr_POD_normal,tpr_POD_normal,'g','LineWidth',2)


hold off;






