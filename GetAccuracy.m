function recAccuracy = GetAccuracy(outputLabel)
load testLabel;
correctP=0;
if isempty(outputLabel)
    recAccuracy = 0;
else
    for i=1:size(testLabel,1)
        if strcmp(outputLabel(i,:),testLabel(i,:)) % compare the two strings
            correctP=correctP+1;
        end
    end
    recAccuracy=correctP/size(testLabel,1)*100;  % Recognition accuracy
end

