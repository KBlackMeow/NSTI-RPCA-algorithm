tem = imread('./0000.jpg');
[x,y]=size(tem);
imagenames = dir(fullfile('./output1/','*.jpg'));
[length,no] = size(imagenames);
ret = zeros(length,1);
for i=1:length
    img = imread(['./output1/' imagenames(i).name]);
    ret(i)=psnr(tem,img);
end
ret