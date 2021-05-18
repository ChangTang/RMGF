clear all;
close all,clc;
addpath(genpath(cd));
dataPath = 'HyperData';

% method = {'KNN','SVM','LDA'}
method = {'SVM'};

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
% smoothing
for i=1:size(img_src,3);
    img_src(:,:,i) = imfilter(img_src(:,:,i),fspecial('average'));
end

SegPara = 2000;
Ratio = 0.0664; % 
% superpixel segmentation
[labels, Imgpca]= cubseg(img_src,SegPara*Ratio);
img=[];
for i = 1 : L
    img(:,:,i) = img_src(:,:,i) / max(max(img_src(:,:,i)));
end

%  Caculate similarity matrices
knn = 16; %16; Adjust this parameter can get better results
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
    SimMatrix{SuperIndex} = convert2sim_knn(TempDis, knn, sigma);
end
%% fusion with diffusion
para.mu = 0.3;
para.max_iter_diffusion = 20;
para.max_iter_alternating = 50;
para.thres = 1e-3;
para.is_sparse = 0;
I = eye(size(SimMatrix{1}), 'single');

para.beta = ones(length(SimMatrix), 1)/length(SimMatrix);
para.lambda = 19;    % weight regularizor
[A, ~, ~] = fusion(SimMatrix, I, para, labels, img);
BandK = [5:5:60];
    ResKNN = [];
    ResSVM = [];
    ResLDA = [];
for iBand = 1:length(BandK);
    K = BandK(iBand); % the number of selected bands K
    disp(['Band is ',num2str(K)]);
    CluRes=PridictLabel(A,K);
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
       
    end
    
end
Res=[ResKNN;ResSVM;ResLDA];
fileName = [data,'_RMGF.mat'];
save(fileName,'Res');






