close all;
clear;
clc;
diary log.txt
tic;
dbpath = '../../data/';
%dbpath = 'C:/Needs/AFLW/data/';
dbfile = 'aflw.sqlite';

%save to specificed location
% croppedFacepath = 'E:/AFLW/data/faces';
croppedNonFacepath = 'E:/AFLW/data/no_faces256';
% croppedNonFacepath = 'C:/Needs/AFLW/data/nonfacesTemp';

%loading image from database
mksqlite('open',fullfile(dbpath,dbfile));
fidQuery = 'SELECT Faces.face_id, FaceImages.filepath, FaceRect.x, FaceRect.y, FaceRect.w, FaceRect.h FROM Faces JOIN FaceImages ON Faces.file_id=FaceImages.file_id  JOIN FaceRect ON Faces.face_id=FaceRect.face_id';
faces = mksqlite(fidQuery);

%sorting files names
[~,index] = sortrows({faces.filepath}.'); 
faces = faces(index(end:-1:1)); 
clear index

iter = size(faces,1);
%iter = 56;

%cropping face and resize to 227x227
numrows = 256;
numcols = 256;
croppedSize = [numrows, numcols];

COUNTER = 0;
skip = 0;
i = 1;
imgName = 0;
DisplayTimeout = 1;
while(i <= iter)
    %number of face and nonface data to be created
    %faceCount = 7 (total 8), nonfaceCount = 800 setting took 76813.311680s to finish with core i7 7gen chip
    nonfaceCount = 30;
    
    %load image location
    imageFilepath = faces(i).filepath;   
    im = imread([dbpath 'flickr/' imageFilepath]);
    
    caseOnebb = 0;    
    scale = 0;
    
    %data argumentation
    [Y, X, ~] = size(im);
    imageArea = Y*X;
    
    flag = i;
    while (flag < iter) && (strcmp(faces(flag).filepath, faces(flag + 1).filepath))
        flag = flag + 1;
    end
   
    if flag > i
        COUNTER = COUNTER + 1;
        masked = flag - i + 1;
        rectGT = zeros(masked, 4);
        %faceCropped = uint8(zeros(227, 227, 3, masked));
        overlapArea = zeros(masked, 1);

        for zz = 1:1:masked
            index = i + zz - 1;
            rectGT(zz,:) = [faces(index).x, faces(index).y, faces(index).w, faces(index).h];
            cropped = imcrop(im,rectGT(zz, :));
                     
%           faceCropped(:, :, :, zz) = imresize(cropped, croppedSize);
%           figure();
%           imshow(faceCropped(:,:,:,zz));
%           imageinfo;

            %true groundTruth
            trueX = faces(index).x;
            trueY = faces(index).y;
    
            if(faces(index).x > X)
                trueX = X;
            end
            if(faces(index).y > Y)
                trueY = Y;
            end
            
            %true size of bounding box
            [height, width, ~] = size(cropped);
            
            rectGT(zz,:) = [trueX*(trueX > 0), trueY*(trueY > 0), width - 1, height - 1];
            rectGTArea = width*height;
            overlapArea(zz) = rectGTArea/imageArea;
        end
        
        [~, index] = sort(overlapArea, 'ascend');  
        overlapArea = overlapArea(index);
        rectGT = rectGT(index,:);
        clear index
       
%        %keep smallest value only
%        rectGT = rectGT(1,:);
       
        %getting the width and height of the smallest size bounding box
        minWidth = rectGT(1,3);
        minHeight = rectGT(1,4);
            
        if overlapArea(masked) >= 0.4
%            skip = skip+1;
%            nonfaceCount = 0;
            scale = 1;
        end 
       
    else
        rectGT = [faces(i).x, faces(i).y, faces(i).w, faces(i).h];
        cropped = imcrop(im,rectGT);
        %faceCropped = imresize(cropped, croppedSize);
        
        %true groundTruth
        trueX = faces(i).x;
        trueY = faces(i).y;
    
        if(faces(i).x > X)
            trueX = X;
        end
        if(faces(i).y > Y)
            trueY = Y;
        end
        
        [height, width, ~] = size(cropped);

        rectGT = [trueX*(trueX > 0), trueY*(trueY > 0), width - 1, height - 1];
        rectGTArea = width*height;
        overlapArea = rectGTArea/imageArea;
        
        if overlapArea >= 0.4
%             skip = skip+1;
%             nonfaceCount = 0;
            scale = 1;
        end 
        caseOnebb = 1;
    end

    %save cropped face, minus 1 to both variable to get 0-based name
    nameNF = 1 + (imgName - skip)*nonfaceCount;
    
    Timeout = 1500000;
    while nonfaceCount > 0
        if caseOnebb
            xRD = round((X - width)*rand);
            yRD = round((Y - height)*rand);
            
            if scale == 1
                factor = rand*0.6+0.4;
            else
                factor = 1;
            end
            
            rectRD = [xRD, yRD, round((width - 1)*factor), round((height - 1)*factor)];
            intersectionArea = rectint(rectGT, rectRD); %in pixels
            unionArea = width*height*2 - intersectionArea; %in pixels
            IOUrate = intersectionArea/unionArea;  %IOU calculation
          
            if (nonfaceCount > 0) && (IOUrate <= 0.3)
                fileName = sprintf('%d.jpg', nameNF);
                fullfileName = fullfile(croppedNonFacepath, fileName);
                nonfaceCropped = imresize(imcrop(im, rectRD), croppedSize);
                imwrite(nonfaceCropped, fullfileName);
                nonfaceCount = nonfaceCount - 1;
                nameNF = nameNF + 1; 
            end
        else
            IOUrate = zeros(masked,1);
            xRD = round((X - minWidth)*rand);
            yRD = round((Y - minHeight)*rand);
            
            if scale == 1
                factor = rand*0.6+0.4;
            else
                factor = 1;
            end
            
            rectRD = [xRD, yRD, round((minWidth - 1)*factor), round((minHeight - 1)*factor)];
            
            for zz = 1:1:masked
                intersectionArea = rectint(rectGT(zz,:), rectRD); %in pixels
                unionArea = width*height*2 - intersectionArea; %in pixels
                IOUrate(zz) = intersectionArea/unionArea;  %IOU calculation
            end
                
            if (nonfaceCount > 0) && (max(IOUrate) <= 0.2)
                fileName = sprintf('%d.jpg', nameNF);
                fullfileName = fullfile(croppedNonFacepath, fileName);
                nonfaceCropped = imresize(imcrop(im, rectRD), croppedSize);
                imwrite(nonfaceCropped, fullfileName);
                nonfaceCount = nonfaceCount - 1;
                nameNF = nameNF + 1; 
            end
        end
        
        Timeout = Timeout - 1;
        if Timeout <= 0
            skip = skip + 1;
            nonfaceCount = 0;
            Timeout = 1500000;
            
            fprintf('Time out ! Skip this image: %s\n', imageFilepath);
        end   
        
        DisplayTimeout = DisplayTimeout - 1;
        if DisplayTimeout <= 0
            DisplayTimeout = 10000000;
            a = dir([croppedNonFacepath, '\*.jpg']);
            numImg = size(a,1); 
            
            fprintf('number of images in folder: %d\n', numImg);
            fprintf('name of latest images in folder: %d\n', nameNF - 1);
            fprintf('expected : %d\n\n', (imgName + 1 - skip) * 850);
            toc;
        end
    end
    
    if caseOnebb
       i = i + 1;
    else
       i = flag + 1;
    end
    
    save ('CurrentState', 'skip', 'i', 'flag', 'nameNF', 'numImg','imgName');
    imgName = imgName + 1;
end
mksqlite('close');
fprintf('number of nonfaces: %d\n', nameNF - 1);
fprintf('time of flags used: %d\n', COUNTER);
fprintf('number of skips: %d\n', skip);

a = dir([croppedNonFacepath, '\*.jpg']);
numImg = size(a,1); 
fprintf('number of images in folder: %d\n', numImg);
expectedfactor = 30;
fprintf('expected : %d\n', (imgName - skip) * expectedfactor);
fprintf('Done !\n');
toc;
diary off