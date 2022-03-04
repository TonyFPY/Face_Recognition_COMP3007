classdef FaceNetFCLayer < nnet.layer.Layer
    properties
    end
    properties (Learnable)
        W;
        b;
    end
    methods
        function layer = FaceNetFCLayer(input_shape, output_shape, name)
            layer.Name = name;
            layer.Description = 'A fully connected layer 128';
            layer.b = ones([1 output_shape]);
            layer.W = ones([input_shape, output_shape]);
        end
        function Z = predict(layer, X)
            Z = X;
        end

    end
end