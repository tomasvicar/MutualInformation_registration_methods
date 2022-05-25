fixed = dicomread('knee1.dcm');
moving = dicomread('knee2.dcm');

imwrite(uint8(mat2gray(double(fixed))*255),'knee1.png')

imwrite(uint8(mat2gray(double(moving))*255),'knee2.png')