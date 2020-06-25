function imgs = imgreader(path,lname)
    imagenames = dir(fullfile(path,lname));
    [length,no] = size(imagenames);
    tem = imread([path imagenames(1).name]);
    [x,y]=size(tem);
    imgs = zeros(length,x*y);
    for i=1:length
        img = imread([path imagenames(i).name]);
        %img = rgb2gray(img);
    	imgs(i,:) = reshape(img,1,[]);
    end
end