close all;
clear;

tic;
dbpath = '../../data/';
%dbpath = 'C:/Needs/AFLW/data/';
dbfile = 'aflw.sqlite';

%save to specificed location
 croppedFacepath = 'E:/AFLW/data/face256';
% croppedNonFacepath = 'E:/AFLW/data/nonfaces';
% croppedFacepath = 'C:/Needs/AFLW/data/faces';
%croppedNonFacepath = 'C:/Needs/AFLW/data/nonfaces';

%loading image from database
mksqlite('open',fullfile(dbpath,dbfile));
fidQuery = 'SELECT Faces.face_id, FaceImages.db_id, FaceImages.filepath, FaceRect.x, FaceRect.y, FaceRect.w, FaceRect.h FROM Faces JOIN FaceImages ON Faces.file_id=FaceImages.file_id  JOIN FaceRect ON Faces.face_id=FaceRect.face_id';
faces = mksqlite(fidQuery);

iter = size(faces,1);

%cropping face and resize to 256*256
numrows = 256;
numcols = 256;
croppedSize = [numrows, numcols];

i = 1;
while(i <= iter)
    %cutting face in each image
    %face_id = faces(i).face_id;

    %number of face and nonface data to be created
    %faceCount = 7 (total 8), nonfaceCount = 800 setting took 76813.311680s to finish with core i7 7gen chip
    faceCount = 9; %the total images are always +1;
    nonfaceCount = 0;

    %load image location
    imageFilepath = faces(i).filepath;
    imageDb_id = faces(i).db_id;        
    
    im = imread([dbpath imageDb_id '/' imageFilepath]);

    rectGT = [faces(i).x, faces(i).y, faces(i).w, faces(i).h];
    cropped = imcrop(im,rectGT);
    faceCropped = imresize(cropped, croppedSize);
    % figure;
    % imshow(faceCropped);

    %save cropped face, minus 1 to both variable to get 0-based name
    nameF = i+(i-1)*faceCount;
    nameNF = 1+(i-1)*nonfaceCount;

    fileName = sprintf('%d.jpg',nameF);
    fullfileName = fullfile(croppedFacepath,fileName);
    imwrite(faceCropped, fullfileName);

    %data argumentation
    [Y, X, ~] = size(im);
    
    %true size of bounding box
    [height, width, ~] = size(cropped);
    
    %true groundTruth
    trueX = faces(i).x;
    trueY = faces(i).y;
    
    if(faces(i).x > X)
       trueX = X;
    end
    if(faces(i).y > Y)
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
                if overlapRate >= 0.5
                    nameF = nameF + 1;
                    fileName = sprintf('%d.jpg', nameF);
                    fullfileName = fullfile(croppedFacepath,fileName); 
                    faceCropped = imresize(imcrop(im, rectRD), croppedSize);
                    imwrite(faceCropped, fullfileName);
                    faceCount = faceCount - 1;
                end
            end
            
            if nonfaceCount > 0
                if overlapRate1 >= 0.3
                    fileName = sprintf('%d.jpg', nameNF);
                    fullfileName = fullfile(croppedNonFacepath,fileName);
                    blank = zeros(croppedSize);
                    imwrite(blank, fullfileName);               
                    nonfaceCount = nonfaceCount - 1;
                    nameNF = nameNF+1;
                else
                    if overlapRate <= 0.3 
                        fileName = sprintf('%d.jpg', nameNF);
                        fullfileName = fullfile(croppedNonFacepath,fileName);
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
mksqlite('close');
toc;
