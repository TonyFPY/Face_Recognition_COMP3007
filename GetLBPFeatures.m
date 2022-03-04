function imglbp = GetLBPFeatures(img,radius,neighbors)
imgSize = size(img);
if numel(imgSize) > 2
    imgG = rgb2gray(img);
else
    imgG = img;
end
[rows, cols] = size(imgG);
rows=int16(rows);
cols=int16(cols);
imglbp = uint8(zeros(rows-2*radius, cols-2*radius));

for k=0:neighbors-1
    % deviation rx，ry  according to the center point        
    rx = radius * cos(2.0 * pi * k / neighbors);
    ry = -radius * sin(2.0 * pi * k / neighbors);
    % round    
    x1 = floor(rx);
    x2 = ceil(rx);
    y1 = floor(ry);
    y2 = ceil(ry);
    % project them into 0-1        
    tx = rx - x1;
    ty = ry - y1;
    % calculate the weights based on deviation
    w1 = (1-tx) * (1-ty);
    w2 = tx * (1-ty);
    w3 = (1-tx) * ty;
    w4 = tx * ty;

    for i=radius+1:rows-radius
        for j=radius+1:cols-radius
            center = imgG(i, j);
            % obtain the kth pixel value via bilinear interpolation               
            neighbor = imgG(i+x1, j+y1)*w1 + imgG(i+x1, j+y2)*w2 + imgG(i+x2, j+y1)*w3 + imgG(i+x2, j+y2)*w4;
            % obtain the LBP features
            if neighbor > center
                flag = 1;
            else
                flag = 0;
            end
            imglbp(i-radius, j-radius) = bitor(imglbp(i-radius, j-radius), bitshift(flag, neighbors-k-1));
        end
    end
end
end

