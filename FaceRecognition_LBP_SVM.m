function outputLabel = FaceRecognition_LBP_SVM(trainPath, testPath, data)
%%   A face reconition method using Circular LBP.
%    trainPath - directory that contains the given training face images
%    testPath  - directory that constains the test face images
%    outputLabel - predicted face label for all tested images 

%% Retrieve training images and labels
folderNames = dir(trainPath);
folderNames = folderNames(~startsWith({folderNames.name}, '.')); % exclude '.', '..', '.DS_Store'

if data == "Augmentation"
    classSize = 4; 
else
    classSize = 1;
end
trainingSize = length(folderNames) * classSize;
imgSize = 600;
trainImgSet = zeros(imgSize,imgSize,3,trainingSize); % all images are 3 channels with size of 600x600
labelImgSet = char(zeros([trainingSize, 6]));

for i = 1:length(folderNames)
    imgName = dir([trainPath,folderNames(i,:).name,'/*.jpg']);
    imgPath = [trainPath, folderNames(i,:).name, '/', imgName.name];
    
    trainImgSet(:,:,:,(i-1)*classSize+1:i*classSize) = AugmentData(imgPath,classSize,imgSize);
    for j = 1 : classSize
        labelImgSet((i-1)*classSize + j,:) = folderNames(i,:).name;
    end
end

%% Get the size of the feature
radius = 8;
neighbors = 8;
faceSize = 72;

face = rgb2gray(uint8(trainImgSet(:,:,:,1)));
% face = FaceDetector(face);
face = imresize(face, [faceSize, faceSize]);

featureFace = GetLBPFeatures(face,radius,neighbors);
[w,h] = size(featureFace);
feature = double(featureFace(:))'; % normalise the intensity to 0-1& store the feature vector
feature = (feature-mean(feature))/std(feature); % Use zero-m
featuresMatrix = zeros(trainingSize,w*h,'single'); %构造矩阵

% face = double(face(:))/255'; % normalise the intensity to 0-1& store the feature vector
% face = (face-mean(face))/std(face); % Use zero-m
% face = reshape(face,faceSize,faceSize);
% feature = extractLBPFeatures(face,'Radius',8);%获取feature的大小
% featuresMatrix = zeros(trainingSize,size(feature,2),'single'); %构造矩阵

featuresMatrix(1,:) = feature;

%% Compute and extract features
for i = 2 : trainingSize
    face = rgb2gray(uint8(trainImgSet(:,:,:,i)));
%     face = FaceDetector(face);
    face = imresize(face, [faceSize, faceSize]);
    
    featureFace = GetLBPFeatures(face,radius,neighbors);
    feature = double(featureFace(:))'; % normalise the intensity to 0-1& store the feature vector
    feature = (feature-mean(feature))/std(feature); % Use zero-m

%     face = double(face(:))/255'; % normalise the intensity to 0-1& store the feature vector
%     face = (face-mean(face))/std(face); % Use zero-m
%     face = reshape(face,faceSize,faceSize);
%     feature = extractLBPFeatures(face,'Radius',8);%获取feature的大小

    featuresMatrix(i,:) = feature;
end

%% Train the SVM model
classifier = fitcecoc(featuresMatrix, labelImgSet, 'Coding', 'onevsall');

%% Predict the label with cosine similarity
testImgNames=dir([testPath,'*.jpg']);
outputLabel = char(zeros([size(testImgNames, 1), 6]));
for i=1:size(testImgNames,1)
    % pre-process the test image
    testImg = imread([testPath, testImgNames(i,:).name]);
    testImg = imresize(testImg, [imgSize, imgSize]);
    testImg = rgb2gray(uint8(testImg));
%     testImg = FaceDetector(testImg);
    testFace = imresize(testImg, [faceSize, faceSize]);
    
    testFaceFeature = GetLBPFeatures(testFace,radius,neighbors);
    testFaceFeature = double(testFaceFeature(:))'; % normalise the intensity to 0-1& store the feature vector
    testFaceFeature = (testFaceFeature-mean(testFaceFeature))/std(testFaceFeature); % Use zero-m
   
%     testFace=double(testFace(:))/255';                % normalise the intensity to 0-1& store the feature vector
%     testFace=(testFace-mean(testFace))/std(testFace);
%     testFace = reshape(testFace,faceSize,faceSize);
%     testFaceFeature = extractLBPFeatures(testFace,'Radius',4);
   
    outputLabel(i,:) = predict(classifier, testFaceFeature);  
end
end