function outputLabel = FaceRecognition_HOG_COS(trainPath,testPath,data)
%%   A face reconition method using Histogram of Oriented Gradient.
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
imgSize = 224;
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
faceSize = 128;
cellSize = [10 10];
blockSize = [5 5];

face = rgb2gray(uint8(trainImgSet(:,:,:,1)));
face = FaceDetector(face);
face = imresize(face, [faceSize, faceSize]);

face = double(face(:))/255'; % normalise the intensity to 0-1& store the feature vector
face = (face-mean(face))/std(face); % Use zero-m
face = reshape(face,faceSize,faceSize);

feature = extractHOGFeatures(face,'CellSize',cellSize,'BlockSize',blockSize);%获取feature的大小
featuresMatrix = zeros(trainingSize,size(feature,2),'single'); %构造矩阵
featuresMatrix(1,:) = feature;

%% Compute and extract features

for i = 2 : trainingSize
    face = rgb2gray(uint8(trainImgSet(:,:,:,i)));
    face = FaceDetector(face);
    face = imresize(face, [faceSize, faceSize]);

    face = double(face(:))/255'; % normalise the intensity to 0-1& store the feature vector
    face = (face-mean(face))/std(face); % Use zero-m
    face = reshape(face,faceSize,faceSize);

    feature = extractHOGFeatures(face,'CellSize',cellSize,'BlockSize',blockSize);%获取feature的大小
    featuresMatrix(i,:) = feature;
end

%% Predict the label with cosine similarity
testImgNames=dir([testPath,'*.jpg']);
outputLabel=char(zeros([trainingSize, 6]));
for i=1:size(testImgNames,1)
    % pre-process the test image
    testFace = imread([testPath, testImgNames(i,:).name]);
    testFace = imresize(testFace, [imgSize, imgSize]);
    testFace = rgb2gray(uint8(testFace));
    testFace = FaceDetector(testFace);
    testFace = imresize(testFace, [faceSize, faceSize]);

    testFace=double(testFace(:))/255';                % normalise the intensity to 0-1& store the feature vector
    testFace=(testFace-mean(testFace))/std(testFace);
    testFace = reshape(testFace,faceSize,faceSize);

    featuresTestImg = extractHOGFeatures(testFace,'CellSize',cellSize,'BlockSize',blockSize);
   
    r = zeros(trainingSize,1);
    for j = 1 : trainingSize
        A = featuresMatrix(j,:);
        B = featuresTestImg;
%         temp = corr2(A,B);

        A = A - mean2(A);
        B = B - mean2(B);
        temp = sum(sum(A.*B))/sqrt(sum(sum(A.*A))*sum(sum(B.*B)));
        r(j,:) = temp;
    end

    [dist , id] = max(r);
    outputLabel(i,:) = labelImgSet(id,:);
end
end

