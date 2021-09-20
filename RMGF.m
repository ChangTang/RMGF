function [A] = RMGF(SimMatrix, labels, img, Imgpca)
spNO = max(labels(:));
L = size(SimMatrix{1},1);
meanvals = zeros(1,spNO);
meanxy = zeros(spNO,2);
for i = 1:spNO
    SimMatrix{i} = SimMatrix{i}-eye(L);
    inds{i}=[];
    inds{i}=find(labels==i);
    [xi, yi] = find(labels==i);
    meanxy(i,1) = mean(xi);
    meanxy(i,2) = mean(yi);
    meanvals(i) = mean(Imgpca(inds{i}));
end

SpectralDistance = zeros(spNO,spNO);
SpatialDsitance = zeros(spNO,spNO);
for spi = 1:spNO
    for spj = 1:spNO
        %         disp(spi);
        %         disp(spj);
        SpectralDistance(spi,spj) = exp(-0.5*(meanvals(spi)-meanvals(spj)).^2/10000) ;
        SpatialDsitance(spi,spj) = exp(-0.5*((meanxy(spi,1)-meanxy(spj,1)).^2+(meanxy(spi,2)-meanxy(spj,2)).^2)/10000) ;
    end
end
SpectralDistanceSum = sum(SpectralDistance(:));
SpatialDsitanceSum = sum(SpatialDsitance(:));

max_iter = 10;

for iter = 1:max_iter
    disp(['Iter: ', num2str(iter)]);
    for spi = 1:spNO        
        Di = diag(sum(SimMatrix{spi}));
        Ai = Di^(-0.5)*SimMatrix{spi}*Di^(-0.5);
        SimMatrixSum = zeros(L,L);
        for spj = 1:spNO
            if spj~=spi
                SimMatrixSum = SimMatrixSum + Ai*(SpectralDistance(spi,spj)/SpectralDistanceSum + SpatialDsitance(spi,spj)/SpatialDsitanceSum)*SimMatrix{spj}*Ai';
            end
        end
        SimMatrixSum = SimMatrixSum/(spNO-1);
        SimMatrix{spi} = 0.7*SimMatrixSum +0.3*SimMatrix{spi};
    end
    
end

SimMatrixSum = zeros(L,L);
for sp = 1:spNO
    SimMatrixSum = SimMatrixSum + SimMatrix{sp};
end
A = SimMatrixSum/spNO;
t=5;