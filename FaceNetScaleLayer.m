classdef FaceNetScaleLayer < nnet.layer.Layer % & nnet.layer.Formattable (Optional) 

    properties
        Ratio
    end

    properties (Learnable)
    end
    
    methods
        function layer = FaceNetScaleLayer(ratio,numInputs,name)
            layer.Name = name;
            layer.NumInputs = numInputs;
            layer.Description = ...
                "scale the features with" + ratio;
            layer.Type = "Scale Sum";
            layer.Ratio = [1,ratio];
        end
        
        function Z = predict(layer, varargin)
            X = varargin;
            W = layer.Ratio;
            
            % Initialize output
            X1 = X{1};
            sz = size(X1);
            Z = zeros(sz,'like',X1);
            
            % Weighted addition
            for i = 1:layer.NumInputs
                Z = Z + W(i)*X{i};
            end
        end
    end
end