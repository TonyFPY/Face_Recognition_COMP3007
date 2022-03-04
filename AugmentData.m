function imgAug = AugmentData(imgPath,classSize,imgSize)
    imgAug = zeros(imgSize,imgSize,3,classSize,'uint8');
    img1 = imread(imgPath);
    img1 = imresize(img1, [imgSize, imgSize]);
    imgAug(:,:,:,classSize) = img1;
    numOfAugMethods = 3;
    
    if(classSize > 1)
        for i = classSize-1 : -1 : 1
            switch(mod(i,numOfAugMethods))
%                 case 1
%                     newFace = translate(img1);
%                 case 2
%                     newFace = rotate(img1);                
%                 case 3
%                     newFace = enhanceContrast(img1);
%                 case 4
%                     newFace = addGaussianNoise(img1);
%                 case 5 
%                     newFace = invertImg(img1);
%                 case 6
%                     newFace = brighten(img1);
%                 case 7 
%                     newFace = darken(img1);
%                 otherwise
%                     newFace = flipImg(img1);
                case 1
                    newFace = rotate(img1);                
                case 2
                    newFace = addGaussianNoise(img1);
                otherwise
                    newFace = flipImg(img1);
            end
            imgAug(:,:,:,i) = newFace;
        end
    end
    
    function newImg = enhanceContrast(img)
        newImg = imadjust(img, [0.1,1], []);
    end

    function newImg = addGaussianNoise(img)
        newImg = imnoise(img,'gaussian',0.1,0.02);  
    end

    function newImg = flipImg(img)
        newImg = flip(img,2);
    end

    function newImg = rotate(img)
        degree = -10 + round(rand(1)*20);
        newImg = imrotate(img,degree,'bilinear','crop');
    end

    function newImg = translate(img)
        [row,col,c] = size(img);
        ratio1 = 1/(15 + round(rand(1)*5));
        ratio2 = 1/(15 + round(rand(1)*5));
        n1 = (-1)^(round(rand(1)));
        n2 = (-1)^(round(rand(1)));
        devX = col * ratio1 * n1;
        devY = row * ratio2 * n2;
        newImg = imtranslate(img,[devX, devY]);
    end

    % up
    function newImg = translateUp(img)
        [row,col,c] = size(img);
        ratio = 1/(15 + round(rand(1)*5));
        devY = row * ratio;
        newImg = imtranslate(img,[0, devY]);
    end

    % down
    function newImg = translateDown(img)
        [row,col,c] = size(img);
        ratio = 1/(15 + round(rand(1)*5));
        devY = row * ratio * (-1);
        newImg = imtranslate(img,[0, devY]);
    end
    
    % right
    function newImg = translateRight(img)
        [row,col,c] = size(img);
        ratio = 1/(15 + round(rand(1)*5));
        devX = col * ratio;
        newImg = imtranslate(img,[devX, 0]);
    end
    
    % left
    function newImg = translateLeft(img)
        [row,col,c] = size(img);
        ratio = 1/(15 + round(rand(1)*5));
        devX = col * ratio * (-1);
        newImg = imtranslate(img,[devX, 0]);
    end

    function newImg = changeColor(img)
        channel = 1 + round(rand(1)*2);
        img(:,:,channel) = 0;
        newImg = img;
    end

    function newImg = invertImg(img)
        newImg = imcomplement(img);
    end

    function newImg = brighten(img)
        val = 10 + round(rand(1)*10);
        newImg = img + val;
    end

    function newImg = darken(img)
        val = 10 + round(rand(1)*10);
        newImg = img - val;
    end

    
    % scale 1
    
    % scale 2
end

