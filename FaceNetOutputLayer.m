classdef FaceNetOutputLayer < nnet.layer.RegressionLayer
    methods
        function layer = FaceNetOutputLayer(name)
            layer.Name = name;
            layer.Description = 'Output a feature vector 1*128';
        end
        
        function feature = forwardLoss(layer, Y, T)
            feature = Y;
        end
    end
end