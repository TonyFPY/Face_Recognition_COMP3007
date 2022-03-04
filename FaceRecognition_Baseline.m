function outputLabel = FaceRecognition_Baseline(trainPath,testPath)
%%   A simple face reconition method using cross-correlation based tmplate matching.
%    trainPath - directory that contains the given training face images
%    testPath  - directory that constains the test face images
%    outputLabel - predicted face label for all tested images 

%% Retrieve training images and labels
folderNames=dir(trainPath);
folderNames=folderNames(~startsWith({folderNames.name}, '.')); % exclude '.', '..', '.DS_Store'

trainImgSet=zeros(600,600,3,length(folderNames)); % all images are 3 channels with size of 600x600
labelImgSet=folderNames; % the folder names are the labels
for i = 1:length(folderNames)
    imgName=dir([trainPath,folderNames(i,:).name,'/*.jpg']);
    trainImgSet(:,:,:,i)= imread([trainPath, folderNames(i,:).name, '/', imgName.name]);
end

%% Prepare the training image: Here we simply use the gray-scale values as template matching. 
% You should implement your own feature extraction/description method here
trainTmpSet=zeros(600*600,size(trainImgSet,4)); % use 600x600 feature vector 
for i=1:size(trainImgSet,4)
    tmpI= rgb2gray(uint8(trainImgSet(:,:,:,i)));
%     tmpI=reshape(tmpI',600*600,1);
%     tmpI=double(tmpI);
    tmpI= double(tmpI(:))/255'; % normalise the intensity to 0-1& store the feature vector
    trainTmpSet(:,i)=(tmpI-mean(tmpI))/std(tmpI); % Use zero-mean normalisation. This is not neccessary if you use other gradient-based feature descriptor
end

%% Face recognition for the test images
testImgNames=dir([testPath,'*.jpg']);
outputLabel=[];
for i=1:size(testImgNames,1)
    % pre-process the test image
    testImg=imread([testPath, testImgNames(i,:).name]);
    tmpI=rgb2gray(uint8(testImg));
%     tmpI=reshape(tmpI',600*600,1);
%     tmpI=double(tmpI);
    tmpI=double(tmpI(:))/255';                % normalise the intensity to 0-1& store the feature vector
    tmpI=(tmpI-mean(tmpI))/std(tmpI);
    
    % cross correlation
    ccValue=trainTmpSet'*tmpI;                % perform dot product (cross correlation with all the training images, and
    labelIndx=find(ccValue==max(ccValue));    % retrieve the label that correspondes to the largest value. 
    outputLabel=[outputLabel;labelImgSet(labelIndx(1),:).name];   % store the outputLabels for each of the test image
end

