clear all;
close all;
trainPath='./FaceDatabase/Train/'; 
testPath='./FaceDatabase/Test/';
%% Start
disp('------------------------ Start ---------------------------');

%% Choose whether to use data augmentation in the programme

dataAug = "Augmentation"; % "Augmentation" "None" 

%% Baseline Method - Face Recognition by Intensity-based Template Matching
% Time = 28.5992
% Accuracy = 25.3720%

tic;
   outputLabel = FaceRecognition_Baseline(trainPath, testPath);
baseLineTime = toc;
recAccuracy = GetAccuracy(outputLabel);
s = ['Baseline: time = ' ,num2str(baseLineTime),', acc = ', num2str(recAccuracy)];
disp(s)

%% Method 1 - FaceNet + SVM

tic;
  outputLabel1 = FaceRecognition_FACENET(trainPath, testPath); 
method1Time = toc;
recAccuracy = GetAccuracy(outputLabel1);
s = ['FACENET : time = ' ,num2str(method1Time),', acc = ', num2str(recAccuracy)];
disp(s)

%% Show whether the data augmentation is selected

disp('**********************************************************');
if dataAug == "None"
    disp('* You are running the programs without data augmentation *');
else
    disp('* You are running the programs with data augmentation.   *');
end
disp('**********************************************************');

%% Method 2 - HOG + SVM

tic;
  outputLabel2 = FaceRecognition_HOG_SVM(trainPath, testPath, dataAug); % "augmentation"
method2Time = toc;
recAccuracy = GetAccuracy(outputLabel2);
s = ['HOG+SVM : time = ' ,num2str(method2Time),', acc = ', num2str(recAccuracy)];
disp(s)

%% Method 3 - LBP + SVM

tic;
  outputLabel3 = FaceRecognition_LBP_SVM(trainPath, testPath, dataAug); % "augmentation"
method3Time = toc;
recAccuracy = GetAccuracy(outputLabel3);
s = ['LBP+SVM : time = ' ,num2str(method3Time),', acc = ', num2str(recAccuracy)];
disp(s)

%% Method 4 - Eigenface + SVM

tic;
  outputLabel4 = FaceRecognition_EIG_SVM(trainPath, testPath, dataAug); % "augmentation"
method4Time = toc;
recAccuracy = GetAccuracy(outputLabel4);
s = ['EIG+SVM : time = ' ,num2str(method4Time),', acc = ', num2str(recAccuracy)];
disp(s)

%% Method 5 - HOG + Cosine Similarity

tic;
  outputLabel5 = FaceRecognition_HOG_COS(trainPath, testPath, dataAug); % "augmentation"
method5Time = toc;
recAccuracy = GetAccuracy(outputLabel5);
s = ['HOG+COS : time = ' ,num2str(method5Time),', acc = ', num2str(recAccuracy)];
disp(s)

%% Method 6 - LBP + Cosine Similarity

tic;
  outputLabel6 = FaceRecognition_LBP_COS(trainPath, testPath, dataAug); % "augmentation"
method6Time = toc;
recAccuracy = GetAccuracy(outputLabel6);
s = ['LBP+COS : time = ' ,num2str(method6Time),', acc = ', num2str(recAccuracy)];
disp(s)

%% Method 7 - Eigenface + Cosine Similarity

tic;
  outputLabel7 = FaceRecognition_EIG_COS(trainPath,testPath, dataAug); % 
method7Time = toc;
recAccuracy = GetAccuracy(outputLabel7);
s = ['EIG+COS : time = ' ,num2str(method7Time),', acc = ', num2str(recAccuracy)];
disp(s)

%% Finish
disp('------------------------ Finish --------------------------');

