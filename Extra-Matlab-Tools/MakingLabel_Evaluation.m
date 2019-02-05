tic;
%Load LabelPara
%manually insert all data locations
facePath = 'E:/AFLW/data/faceEvaluationTraining';
nonfacePath  = 'E:/AFLW/data/nofaceEvaluationTraining';
%paths = cell(length(properties('Labels')),1);

testsetfacePath = 'E:/AFLW/data/EvaluationTestingFace';
testsetnofacePath = 'E:/AFLW/data/EvaluationTestingnoFace';

%store all path above into one cell, manually modify  it corresponding number
%of Labels and Paths
paths = {'faceEvaluationTraining'; 'nofaceEvaluationTraining'};
paths2 = {'EvaluationTestingFace'; 'EvaluationTestingnoFace'};

%Commenting out from here
% % filesNamef = dir([facePath, '\*.jpg']);
% % filesNamenf = dir([nonfacePath, '\*.jpg']);
% % 
% % toc;
% % batch = 128; %upon the batch of your training network
% % ratio = 1/4; %possitive / negative
% % 
% % positive = round(batch*ratio);
% % negative = batch - positive;
% % 
% % a = length(filesNamef);
% % b = length(filesNamenf);
% % labelBreak = [a, b];
% % 
% % toc;
% % 
% % aMat = (1:1:a);
% % bMat = (1:1:b);
% % 
% % trnSet = [];
% % 
% % while(a >= positive)
% %     trnIndexPos = randperm(a, positive);
% %     trnIndexNeg = randperm(b, negative);
% %     oneBatch = [aMat(trnIndexPos) (labelBreak(1) + bMat(trnIndexNeg))];
% %     oneBatch = oneBatch(randperm(length(oneBatch)));
% %     trnSet = [trnSet oneBatch];
% %     aMat(trnIndexPos) = [];
% %     bMat(trnIndexNeg) = [];
% %     a = length(aMat);
% %     b = length(bMat);
% % end
% % 
% % oneBatch = [aMat, labelBreak(1) + bMat];
% % oneBatch = oneBatch(randperm(length(oneBatch)));
% % trnSet = [trnSet oneBatch];
% % 
% % trnID = fopen('trainEvaluation.txt','w');
% % write_to_file(trnID, trnSet, paths, labelBreak);
% % fclose(trnID);

%testSet
filesNamef = dir([testsetfacePath, '\*.jpg']);
filesNamenf = dir([testsetnofacePath, '\*.jpg']);
a = length(filesNamef);
b = length(filesNamenf);
labelBreak = [a, b];

aMat = (1:1:a);
bMat = (1:1:b);

tstSet = [aMat, labelBreak(1) + bMat];
tstSet = tstSet(randperm(length(tstSet)));
tstID = fopen('testEvaluation.txt','w');
write_to_file(tstID, tstSet, paths2, labelBreak);
fclose(trnID);

disp('done');
%save('LabelPara');
%save(FilesNamenf,'filesNamenf','-v7.3');
toc;