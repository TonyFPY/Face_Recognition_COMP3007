function face = FaceDetector(img)

% initialise a face detector
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART'); % FrontalFaceCART FrontalFaceLBP

% acquire bounding box of the face
bbox = faceDetector(img); % [x y width height]

% %Annotate detected faces.
% IFaces = insertObjectAnnotation(img,'rectangle',bbox,'Face');   
% figure
% imshow(IFaces)

face = img;
threshold = 100; %100
if not(isempty(bbox))
    if (size(bbox,1) == 1)
        w = bbox(1,3);
        h = bbox(1,4);
        if(w >= threshold && h >= threshold)
            face = imcrop(img, bbox);
        end 
    else
        [val, id] = max(bbox(:,3));
        w = bbox(id,3);
        h = bbox(id,4);
        if(w >= threshold && h >= threshold)
            face = imcrop(img, bbox(id,:));
        end 
    end
end

end
