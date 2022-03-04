function outputLabel = FaceRecognition_EIG_COS(trainPath,testPath,data)
%%   A face reconition method using Eigenfaces.
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
trainImgSet = zeros(imgSize,imgSize,3,trainingSize,'uint8'); % all images are 3 channels with size of 600x600
labelImgSet = char(zeros([trainingSize, 6]));

for i = 1:length(folderNames)
    imgName = dir([trainPath,folderNames(i,:).name,'/*.jpg']);
    imgPath = [trainPath, folderNames(i,:).name, '/', imgName.name];
    
    trainImgSet(:,:,:,(i-1)*classSize+1:i*classSize) = AugmentData(imgPath,classSize,imgSize);
    for j = 1 : classSize
        labelImgSet((i-1)*classSize + j,:) = folderNames(i,:).name;
    end
end

%% Prepare the training image
faceSize = 56;
trainTmpSet = zeros(faceSize*faceSize,trainingSize,'double'); % use 600x600 feature vector 
for i = 1:trainingSize
    face = rgb2gray(uint8(trainImgSet(:,:,:,i)));
    face = FaceDetector(face);
    face = imresize(face, [faceSize, faceSize]); 
    
    face= double(face(:))/255'; % normalise the intensity to 0-1& store the feature vector
    trainTmpSet(:,i) = (face-mean(face))/std(face); % Use zero-mean normalisation. This is not neccessary if you use other gradient-based feature descriptor
end

%% Compute the Eigen faces, mean face and features
numOfClass = length(folderNames);
total = trainingSize;
ratio = 0.985;

meanFace = mean(trainTmpSet,2); % compute a 'mean' face of the whole training set
dev = trainTmpSet - repmat(meanFace,1,total);

A = dev'*dev; % get covariance matrix
[V D] = eig(A); % extract features from A by SVD

eigenVec = [];
for i = 1 : round(size(V,2)*ratio)
    eigenVec = [eigenVec V(:,i)];
end

eigenFaces = dev * eigenVec;

%% Project the training images into Eigenfaces space. 
projectedImg= [];
for i = 1 : total
    temp = eigenFaces' * dev(:,i); % Projection of centered images into facespace
    projectedImg = [projectedImg temp]; 
end

%% Project the testing images into Eigenfaces space and find the index of image who gets minmum Euclidean distances.
testImgNames=dir([testPath,'*.jpg']);
outputLabel=char(zeros([trainingSize, 6]));
for i=1:size(testImgNames,1)
    % pre-process the test image
    testImg=imread([testPath, testImgNames(i,:).name]);
    testImg = imresize(testImg, [imgSize, imgSize]);
    testImg = rgb2gray(uint8(testImg));
    testFace = FaceDetector(testImg);
    testFace = imresize(testFace, [faceSize, faceSize]);
    
    testFace=double(testFace(:))/255';                % normalise the intensity to 0-1& store the feature vector
    testFace=(testFace-mean(testFace))/std(testFace);
    
    diff = testFace-meanFace; 
    projectedTestImg = eigenFaces'*diff; % Test image feature vector

    cosine = zeros(total,1);
    for j = 1 : total
        A = projectedImg(:,j);
        B = projectedTestImg;
%         temp = corr2(A,B);
%         a = A(:);
%         b = B(:);
%         temp = dot(a,b) / (sqrt( sum( a.*a )) * sqrt( sum( b.*b )));
        A = A - mean2(A);
        B = B - mean2(B);
        temp = sum(sum(A.*B))/sqrt(sum(sum(A.*A))*sum(sum(B.*B)));
        cosine(j,:) = temp;
    end

    [dist , id] = max(cosine);
    outputLabel(i,:) = labelImgSet(id,:);  
end
end

