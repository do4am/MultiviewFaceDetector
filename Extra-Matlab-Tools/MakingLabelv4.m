tic;
%Load LabelPara
%manually insert all data locations
facePath = 'E:\AFLW\data\face256';
nonfacePath  = 'E:\AFLW\data\no_faces256';
%paths = cell(length(properties('Labels')),1);

%store all path above into one cell, manually modify  it corresponding number
%of Labels and Paths
paths = {'face256'; 'no_faces256'};

filesNamef = dir([facePath, '\*.jpg']);
filesNamenf = dir([nonfacePath, '\*.jpg']);

toc;
batch = 128; %upon the batch of your training network
ratio = 1/4; %possitive / negative

positive = round(batch*ratio);
negative = batch - positive;

a = length(filesNamef);
b = length(filesNamenf);
labelBreak = [a, b];
toc;

aMat = (1:1:a);
bMat = (1:1:b);

remain30 = round(0.3*labelBreak(1));
trnSet = [];

while(a > remain30)
    trnIndexPos = randperm(a, positive);
    trnIndexNeg = randperm(b, negative);
    oneBatch = [aMat(trnIndexPos) (labelBreak(1) + bMat(trnIndexNeg))];
    oneBatch = oneBatch(randperm(length(oneBatch)));
    trnSet = [trnSet oneBatch];
    aMat(trnIndexPos) = [];
    bMat(trnIndexNeg) = [];
    a = length(aMat);
    b = length(bMat);
end

%testSet
tstIndexa = randperm(a, round(0.5*a))';
tstIndexb = randperm(b, round(0.01*b))';
tstSet = [aMat(tstIndexa) (labelBreak(1) + bMat(tstIndexb))];
tstSet = tstSet(randperm(length(tstSet)));

%the remain pictures will be put in validation set
aMat(tstIndexa) = [];
bMat(tstIndexb) = [];
a = length(aMat);
b = length(bMat);
tstIndexb = randperm(b, round(0.01*b))';

valSet = [aMat (labelBreak(1) + bMat(tstIndexb))];
valSet = valSet(randperm(length(valSet)));

trnID = fopen('train256.txt','w');
tstID = fopen('test256.txt','w');
valID = fopen('valid256.txt','w');

write_to_file(trnID, trnSet, paths, labelBreak);
write_to_file(tstID, tstSet, paths, labelBreak);
write_to_file(valID, valSet, paths, labelBreak);

fclose(trnID);
fclose(tstID);
fclose(valID);

disp('done');
%save('LabelPara');
%save(FilesNamenf,'filesNamenf','-v7.3');

toc;