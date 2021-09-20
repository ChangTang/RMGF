clear all;
close all,clc;
addpath(genpath(cd));
dataPath = 'HyperData';

% method = {'KNN','SVM','LDA'};%'RF','ELM'};
method = {'KNN'};

% for imethod = 1:length(method)
%     switch method{imethod}
%         case 'KNN'
%             lambdaa   = [3]; % 3NN
%         case 'SVM'
%             lambdaa   = [10000]; % SVM
%         case 'RF'
%             lambdaa   = [100];
%         case 'ELM'
%             lambdaa   = 2.^[8 10 12 14]; % elm
%     end
% end

data = 'Indian_pines';% PaviaU Salinas_corrected Indian_pines_corrected
%% load data
if strcmp(data,'PaviaU')
    load([dataPath,'\','PaviaU.mat']);
    img_src = paviaU;
    clear paviaU;
    
    load([dataPath,'\','PaviaU_gt.mat']);
    img_gt = paviaU_gt(:);
    clear paviaU_gt;
end

if strcmp(data,'Salinas')
    load([dataPath,'\','Salinas_corrected.mat']);
    img_src = salinas_corrected;
    clear salinas_corrected;
    
    load([dataPath,'\','Salinas_gt.mat']);
    img_gt = salinas_gt(:);
    clear salinas_gt;
end

if strcmp(data,'Indian_pines')
    load([dataPath,'\','Indian_pines_corrected.mat']);
    img_src = indian_pines_corrected;
    clear indian_pines_corrected;
    
    load([dataPath,'\','Indian_pines_gt.mat']);
    img_gt = indian_pines_gt(:);
    clear Indian_pines_gt;
end

[W, H, L]=size(img_src);
img_src = double(img_src);
% img_src = reshape(img_src, W * H, L);
%% band selection
tic
%smoothing
for i=1:size(img_src,3);
    img_src(:,:,i) = imfilter(img_src(:,:,i),fspecial('average'));
end

SegPara = 2000;
Ratio = 0.0664; %  paviaU
% superpixel segmentation
[labels, Imgpca]= cubseg(img_src,SegPara*Ratio);
img=[];
for i = 1 : L
    img(:,:,i) = img_src(:,:,i) / max(max(img_src(:,:,i)));
end

%  Caculate similarity matrices
knn = 16; %16; % You can change this value to get better results
sigma = 0.6;
SuperPixelNo = length(unique(labels))-1;
SimMatrix = cell(1,SuperPixelNo);
for SuperIndex = 1:SuperPixelNo
    disp(['Superpixel: ',num2str(SuperIndex)]);
    TempFea = [];
    TempLabel = find(labels==SuperIndex);
    for BandIndex = 1:L
        TempBand = img(:,:,BandIndex);
        TempFea = [TempFea,TempBand(TempLabel)];
    end
    %     FeaMatrix{SuperIndex} = TempFea;
    TempDis = pdist2(TempFea',TempFea','euclidean'); % n*d;
    SimMatrix{SuperIndex} = bs_convert2sim_knn(TempDis, knn, sigma);
end
%% fusion with diffusion
[A] = RMGF(SimMatrix, labels, Imgpca);
BandK = [5:5:60];
    ResKNN = [];
    ResSVM = [];
    ResLDA = [];
for iBand = 1:length(BandK);
    K = BandK(iBand); % the number of selected bands K
    disp(['Band is ',num2str(K)]);
    CluRes=PridictLabel(A,K);
    % CluRes = SpectralClustering(Sim,K);
    % CluRes = kmeans(img', K, 'emptyaction','singleton'); % data: N*d
    img_src = reshape(img_src, W * H, L);
    img = reshape(img_src, W * H, L);
    Y = SelectBandFromClusRes(CluRes, K, img);
    %% evaluation
    newData = img_src(:, Y)'; % d*n
    trnPer = 0.1; % 10% samples from each class based on selected bands are selected
    clsCnt = length(unique(img_gt)) - 1;
    [trnData, trnLab, tstData, tstLab] = TrainTest(newData', img_gt, trnPer, clsCnt);
    
    tstNum = zeros(1, clsCnt);
    for i = 1 : clsCnt
        index = find(tstLab == i);
        tstNum(i) = length(index);
    end

    for iMethod = 1:length(method)
        if strcmp(method{iMethod},'KNN')
            Predict_label = knnclassify(tstData',trnData',trnLab,3,'euclidean');
            acc = accuracy(tstLab, Predict_label, clsCnt, tstNum);
            ResKNN = [ResKNN,acc];
            t=7;
        end
        if strcmp(method{iMethod},'SVM')
            model = svmtrain(trnLab', trnData', '-s 0 -t 1 -c 0.001 -w1 5 -w-1 0.01');
            [Predict_label, ~,~] = svmpredict(tstLab', tstData',model);    Predict_label =Predict_label';
%             model = libsvmtrain(trnLab', trnData', '-s 0 -t 1 -c 0.001 -w1 5 -w-1 0.01');
%             [Predict_label, accuracyScore, dec_values] = libsvmpredict(tstLab', tstData', model); %Predict_label = Predict_label';
            acc = accuracy(tstLab, Predict_label', clsCnt, tstNum);
            ResSVM = [ResSVM,acc];
        end
        if strcmp(method{iMethod},'LDA')
            obj = ClassificationDiscriminant.fit(trnData', trnLab');
            Predict_label = predict(obj, tstData');
            acc = accuracy(tstLab, Predict_label', clsCnt, tstNum);
            ResLDA = [ResLDA,acc];
            T = 9;
        end
        %         case 'ELM'
        %             [TTrain,TTest,TrainAC,accur_ELM,TY,Predict_label] = elm_kernel([trainlabel_nl' DataTrain_nl],[testlabel' DataTest],1,lambda,'RBF_kernel',1);
        
        %         case 'RF'
        %             Factor = TreeBagger(lambda, DataTrain, trainlabel_nl);
        %             [Predict_label_temp,Scores] = predict(Factor, DataTest);
        %             for ij=1:length(Predict_label_temp); Predict_label(ij) = str2num(Predict_label_temp{ij}); end;
        
    end
    
end
Res=[ResKNN;ResSVM;ResLDA];
fileName = [data,'_RMGF.mat'];
save(fileName,'Res');






