function net = LoadFaceNetModel(modelPath,weightsPath)
%% Inception Resnet V1
% 
%% Import Keras layers into MATLAB
net = importKerasLayers(modelPath, ...
    'ImportWeights',true, ...
    'WeightFile',weightsPath);

%% Create customised layers  
% 5x Block35 scale=0.17
b35Layer1 = FaceNetScaleLayer(0.17,2,'new_Block35_1_ScaleSum');
b35Layer2 = FaceNetScaleLayer(0.17,2,'new_Block35_2_ScaleSum');
b35Layer3 = FaceNetScaleLayer(0.17,2,'new_Block35_3_ScaleSum');
b35Layer4 = FaceNetScaleLayer(0.17,2,'new_Block35_4_ScaleSum');
b35Layer5 = FaceNetScaleLayer(0.17,2,'new_Block35_5_ScaleSum');

% 10x Block17 scale=0.1
b17Layer1 = FaceNetScaleLayer(0.1,2,'new_Block17_1_ScaleSum');
b17Layer2 = FaceNetScaleLayer(0.1,2,'new_Block17_2_ScaleSum');
b17Layer3 = FaceNetScaleLayer(0.1,2,'new_Block17_3_ScaleSum');
b17Layer4 = FaceNetScaleLayer(0.1,2,'new_Block17_4_ScaleSum');
b17Layer5 = FaceNetScaleLayer(0.1,2,'new_Block17_5_ScaleSum');
b17Layer6 = FaceNetScaleLayer(0.1,2,'new_Block17_6_ScaleSum');
b17Layer7 = FaceNetScaleLayer(0.1,2,'new_Block17_7_ScaleSum');
b17Layer8 = FaceNetScaleLayer(0.1,2,'new_Block17_8_ScaleSum');
b17Layer9 = FaceNetScaleLayer(0.1,2,'new_Block17_9_ScaleSum');
b17Layer10 = FaceNetScaleLayer(0.1,2,'new_Block17_10_ScaleSum');

% 6x Block8 scale=0.2
b8Layer1 = FaceNetScaleLayer(0.2,2,'new_Block8_1_ScaleSum');
b8Layer2 = FaceNetScaleLayer(0.2,2,'new_Block8_2_ScaleSum');
b8Layer3 = FaceNetScaleLayer(0.2,2,'new_Block8_3_ScaleSum');
b8Layer4 = FaceNetScaleLayer(0.2,2,'new_Block8_4_ScaleSum');
b8Layer5 = FaceNetScaleLayer(0.2,2,'new_Block8_5_ScaleSum');
b8Layer6 = FaceNetScaleLayer(0.2,2,'new_Block8_6_ScaleSum');

%% Replace placeholder layers that are not supported by MATLAB
net = replaceLayer(net,'Block35_1_ScaleSum',b35Layer1);
net = replaceLayer(net,'Block35_2_ScaleSum',b35Layer2);
net = replaceLayer(net,'Block35_3_ScaleSum',b35Layer3);
net = replaceLayer(net,'Block35_4_ScaleSum',b35Layer4);
net = replaceLayer(net,'Block35_5_ScaleSum',b35Layer5);

net = replaceLayer(net,'Block17_1_ScaleSum',b17Layer1);
net = replaceLayer(net,'Block17_2_ScaleSum',b17Layer2);
net = replaceLayer(net,'Block17_3_ScaleSum',b17Layer3);
net = replaceLayer(net,'Block17_4_ScaleSum',b17Layer4);
net = replaceLayer(net,'Block17_5_ScaleSum',b17Layer5);
net = replaceLayer(net,'Block17_6_ScaleSum',b17Layer6);
net = replaceLayer(net,'Block17_7_ScaleSum',b17Layer7);
net = replaceLayer(net,'Block17_8_ScaleSum',b17Layer8);
net = replaceLayer(net,'Block17_9_ScaleSum',b17Layer9);
net = replaceLayer(net,'Block17_10_ScaleSum',b17Layer10);

net = replaceLayer(net,'Block8_1_ScaleSum',b8Layer1);
net = replaceLayer(net,'Block8_2_ScaleSum',b8Layer2);
net = replaceLayer(net,'Block8_3_ScaleSum',b8Layer3);
net = replaceLayer(net,'Block8_4_ScaleSum',b8Layer4);
net = replaceLayer(net,'Block8_5_ScaleSum',b8Layer5);
net = replaceLayer(net,'Block8_6_ScaleSum',b8Layer6);

%% Add a customised fully connnected layer 
lastLayerName = net.Layers(end).Name;
FCLayer = FaceNetFCLayer(128,128,'fully_connected_layer');
net = addLayers(net,FCLayer);
net = connectLayers(net,lastLayerName,'fully_connected_layer');

%% Add a customised output layer
outputLayer = FaceNetOutputLayer('feature_output_layer');
net = addLayers(net,outputLayer);
net = connectLayers(net,'fully_connected_layer','feature_output_layer');

%% Assemble the customised network
net = assembleNetwork(net);
end

