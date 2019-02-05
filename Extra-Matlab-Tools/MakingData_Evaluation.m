close all;
clear;

tic;
filenameTrain = 'dgstrainPerBB.txt';
facesTrain = importdata(filenameTrain);

filenameTest = 'dgstestPerBB.txt';
testSet = importdata(filenameTest);

%save to specificed location
croppedFacepathTraining = 'E:/AFLW/data/faceEvaluationTraining';
croppedNonFacepathTraining = 'E:/AFLW/data/nofaceEvaluationTraining';
croppedTestsetFacePath = 'E:/AFLW/data/EvaluationTestingFace';
croppedTestsetnoFacePath = 'E:/AFLW/data/EvaluationTestingnoFace';

%loading image from database
prefix_path = '\\GEEKO\project02\Julia2\Dataset';
%path = 'MO04/3305230.jpg';
%fullpath = strcat(prefix_path, '\', path);
% facePath = char(strcat(prefix_path, '\', faces.textdata(1)));
% im = imread(facePath);

%cropping face and resize to 256*256
numrows = 256;
numcols = 256;
croppedSize = [numrows, numcols];

%Commenting out from here
% % iter = length(facesTrain.textdata);
% % i = 1;

% % %Making Training Data
% % while i <= iter
% %     %current setting took 291 seconds to finish. there were 1411 faces
% %     faceCount = 9; %the total images are always +1;
% %     nonfaceCount = 30; %should only be used when the ratio between image and its bb is large
% %     
% %     facePath = char(strcat(prefix_path, '\', facesTrain.textdata(i)));
% %     im = imread(facePath);
% %     faceBB = [facesTrain.data(i, 1), facesTrain.data(i, 3), facesTrain.data(i, 2)-facesTrain.data(i, 1), facesTrain.data(i, 4)-facesTrain.data(i, 3)];
% %     cropped = imcrop(im,faceBB);
% %     faceCropped = imresize(cropped, croppedSize);
% %     
% %     nameF = i+(i-1)*faceCount;
% %     nameNF = 1+(i-1)*nonfaceCount;
% % 
% %     fileName = sprintf('%d.jpg',nameF);
% %     fullfileName = fullfile(croppedFacepathTraining,fileName);
% %     imwrite(faceCropped, fullfileName);
% % 
% %     %data argumentation
% %     [Y, X, ~] = size(im);
% %     
% %     %true size of bounding box
% %     [height, width, ~] = size(cropped);
% %     
% %     %true groundTruth
% %     trueX = facesTrain.data(i, 1);
% %     trueY = facesTrain.data(i, 3);
% %     
% %     if(trueX > X)
% %        trueX = X;
% %     end
% %     if(trueY > Y)
% %        trueY = Y;
% %     end
% %     
% %     rectGT = [trueX*(trueX > 0), trueY*(trueY > 0), width - 1, height - 1];
% %     rectGTArea = width*height;
% %     imageArea = Y*X;
% %     overlapRate1 = rectGTArea/imageArea;
% % 
% %     
% %     while (faceCount > 0) || (nonfaceCount > 0)
% %         xRD = round((X - width)*rand);
% %         yRD = round((Y - height)*rand);
% % 
% %         rectRD = [xRD, yRD, width - 1, height - 1];
% %         intersectionArea = rectint(rectGT, rectRD); %in pixels
% %         unionArea = width*height*2 - intersectionArea; %in pixels
% %         overlapRate = intersectionArea/unionArea;  %IOU calculation
% %         
% %             % 50% <= IOU 
% %             if faceCount > 0
% %                 if overlapRate >= 0.6
% %                     nameF = nameF + 1;
% %                     fileName = sprintf('%d.jpg', nameF);
% %                     fullfileName = fullfile(croppedFacepathTraining,fileName); 
% %                     faceCropped = imresize(imcrop(im, rectRD), croppedSize);
% %                     imwrite(faceCropped, fullfileName);
% %                     faceCount = faceCount - 1;
% %                 end
% %             end
% %             
% %             if nonfaceCount > 0
% %                 if overlapRate1 >= 0.3
% %                     fileName = sprintf('%d.jpg', nameNF);
% %                     fullfileName = fullfile(croppedNonFacepathTraining,fileName);
% %                     blank = zeros(croppedSize);
% %                     imwrite(blank, fullfileName);               
% %                     nonfaceCount = nonfaceCount - 1;
% %                     nameNF = nameNF+1;
% %                 else
% %                     if overlapRate <= 0.3 
% %                         fileName = sprintf('%d.jpg', nameNF);
% %                         fullfileName = fullfile(croppedNonFacepathTraining,fileName);
% %                         nonfaceCropped = imresize(imcrop(im, rectRD), croppedSize);
% %                         imwrite(nonfaceCropped, fullfileName);
% %                         nonfaceCount = nonfaceCount - 1;
% %                         nameNF = nameNF+1;
% %                     end
% %                 end   
% %             end
% %     end
% %     %disp([num2str(i), '.jpg', ' has been created']);
% %     i = i+1;
% % end   

iter = length(testSet.textdata);
i = 1;

%Making Testing Data
while i <= iter
    faceCount = 2; %the total images are always +1;
    nonfaceCount = 5; %should only be used when the ratio between image and its bb is large
    
    facePath = char(strcat(prefix_path, '\', testSet.textdata(i)));
    im = imread(facePath);
    faceBB = [testSet.data(i, 1), testSet.data(i, 3), testSet.data(i, 2)-testSet.data(i, 1), testSet.data(i, 4)-testSet.data(i, 3)];
    cropped = imcrop(im,faceBB);
    faceCropped = imresize(cropped, croppedSize);
    
    nameF = i+(i-1)*(faceCount);
    nameNF = 1+(i-1)*nonfaceCount;
    
    fileName = sprintf('%d.jpg',nameF);
    fullfileName = fullfile(croppedTestsetFacePath,fileName);
    imwrite(faceCropped, fullfileName);

    %data argumentation
    [Y, X, ~] = size(im);
    
    %true size of bounding box
    [height, width, ~] = size(cropped);
    
    %true groundTruth
    trueX = testSet.data(i, 1);
    trueY = testSet.data(i, 3);
    
    if(trueX > X)
       trueX = X;
    end
    if(trueY > Y)
       trueY = Y;
    end
    
    rectGT = [trueX*(trueX > 0), trueY*(trueY > 0), width - 1, height - 1];
    rectGTArea = width*height;
    imageArea = Y*X;
    overlapRate1 = rectGTArea/imageArea;

    
    while (faceCount > 0) || (nonfaceCount > 0)
        xRD = round((X - width)*rand);
        yRD = round((Y - height)*rand);

        rectRD = [xRD, yRD, width - 1, height - 1];
        intersectionArea = rectint(rectGT, rectRD); %in pixels
        unionArea = width*height*2 - intersectionArea; %in pixels
        overlapRate = intersectionArea/unionArea;  %IOU calculation
        
            % 50% <= IOU 
            if faceCount > 0
                if overlapRate >= 0.6
                    nameF = nameF + 1;
                    fileName = sprintf('%d.jpg', nameF);
                    fullfileName = fullfile(croppedTestsetFacePath,fileName); 
                    faceCropped = imresize(imcrop(im, rectRD), croppedSize);
                    imwrite(faceCropped, fullfileName);
                    faceCount = faceCount - 1;
                end
            end
            
            if nonfaceCount > 0
                if overlapRate1 >= 0.3
                    fileName = sprintf('%d.jpg', nameNF);
                    fullfileName = fullfile(croppedTestsetnoFacePath, fileName);
                    blank = zeros(croppedSize);
                    imwrite(blank, fullfileName);               
                    nonfaceCount = nonfaceCount - 1;
                    nameNF = nameNF+1;
                else
                    if overlapRate <= 0.3 
                        fileName = sprintf('%d.jpg', nameNF);
                        fullfileName = fullfile(croppedTestsetnoFacePath, fileName);
                        nonfaceCropped = imresize(imcrop(im, rectRD), croppedSize);
                        imwrite(nonfaceCropped, fullfileName);
                        nonfaceCount = nonfaceCount - 1;
                        nameNF = nameNF+1;
                    end
                end   
            end
    end
    %disp([num2str(i), '.jpg', ' has been created']);
    i = i+1;
end
toc;
% figure;
% imshow(im);
% hold on;
% rectangle('Position',[604, 120, (730-604), (246-120)],'EdgeColor','r')