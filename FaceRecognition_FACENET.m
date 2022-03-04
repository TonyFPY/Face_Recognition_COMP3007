function outputLabel = FaceRecognition_FACENET(trainPath,testPath)
%%   A face reconition method using FaceNet.
%    trainPath - directory that contains the given training face images
%    testPath  - directory that constains the test face images
%    outputLabel - predicted face label for all tested images 

%% Retrieve training images and labels
folderNames = dir(trainPath);
folderNames = folderNames(~startsWith({folderNames.name}, '.')); % exclude '.', '..', '.DS_Store'

classSize = 1;
trainingSize = length(folderNames) * classSize;
imgSize = 200; %200
trainImgSet = zeros(imgSize,imgSize,3,trainingSize,'uint8'); % all images are 3 channels with size of 600x600
labelImgSet = char(zeros([trainingSize, 6]));

for i = 1:length(folderNames)
    imgName = dir([trainPath,folderNames(i,:).name,'/*.jpg']);
    imgPath = [trainPath, folderNames(i,:).name, '/', imgName.name];
    
    img = imread(imgPath);
    trainImgSet(:,:,:,i) = imresize(img, [imgSize, imgSize]);
    labelImgSet(i,:) = folderNames(i,:).name;
end

%% Load FaceNet Model
modelPath = './FaceNet_Model/facenet_keras.h5';
weightsPath = './FaceNet_Model/facenet_keras_weights.h5';
warning('off','all');
facenet = LoadFaceNetModel(modelPath,weightsPath);

%% Get the size of the feature
outputFeatureLength = 128;
featuresMatrix = zeros(trainingSize,outputFeatureLength); 

%% Compute and extract features
faceSize = 160;
for i = 1 : trainingSize
    face = trainImgSet(:,:,:,i);
    face = rgb2gray(uint8(face));
    face = FaceDetector(face);
    face = imresize(face, [faceSize, faceSize]);
    
    face = double(face(:))/255'; % normalise the intensity to 0-1& store the feature vector
    face = (face-mean(face))/std(face); % Use zero-m
    face = reshape(face,faceSize,faceSize);
    face = face(:,:,[1 1 1]);
    
    feature = facenet.predict(face);
    featuresMatrix(i,:) = feature;
end

%% Train the SVM model
classifier = fitcecoc(featuresMatrix, labelImgSet, 'Coding', 'onevsall');

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
    
    testFace = double(testFace(:))/255';                % normalise the intensity to 0-1& store the feature vector
    testFace = (testFace-mean(testFace))/std(testFace);
    testFace = reshape(testFace,faceSize,faceSize);
    testFace = testFace(:,:,[1 1 1]);

    testFeature = facenet.predict(testFace);
   
    outputLabel(i,:) = predict(classifier, testFeature);
end

end

