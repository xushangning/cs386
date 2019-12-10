mkdir('images//1080P//bilinear//');
mkdir('images//1080P//bicubic//');
mkdir('images//1080P//lanczos2//');
mkdir('images//720P//bilinear//');
mkdir('images//720P//bicubic//');
mkdir('images//720P//lanczos2//');
for i=1:200
    ['Current progress: ', num2str(i), ' / 200']
    str = ['images//4K//',num2str(i),'.bmp'];
    I = imread(str);
    %降采样
    R = I(:,:,1);
    G = I(:,:,2);
    B = I(:,:,3);
    R1=R(1:2:end,1:2:end);%每2位采集1位  1080p
    G1=G(1:2:end,1:2:end);
    B1=B(1:2:end,1:2:end);
    I1(:,:,1)=R1;
    I1(:,:,2)=G1;
    I1(:,:,3)=B1;
    R2=R(1:3:end,1:3:end);
    G2=G(1:3:end,1:3:end);
    B2=B(1:3:end,1:3:end);%每3位采集1位 720p
    I2(:,:,1)=R2;
    I2(:,:,2)=G2;
    I2(:,:,3)=B2;
    
    J1 = imresize(I1,[2160,3840], 'bilinear'); %双线性插值
    str1 = ['images//1080P//bilinear//',num2str(i),'.bmp'];
    imwrite(J1, str1);
    K1 = imresize(I1,[2160,3840], 'bicubic');  %双三次插值
    str2 = ['images//1080P//bicubic//',num2str(i),'.bmp'];
    imwrite(K1, str2);
    L1 = imresize(I1,[2160,3840], 'lanczos2'); %Lanczos-2核
    str3 = ['images//1080P//lanczos2//',num2str(i),'.bmp'];
    imwrite(L1, str3);
    
    J2 = imresize(I2,[2160,3840], 'bilinear'); %双线性插值
    str4 = ['images//720P//bilinear//',num2str(i),'.bmp']; 
    imwrite(J2, str4);
    K2 = imresize(I2,[2160,3840], 'bicubic');  %双三次插值
    str5 = ['images//720P//bicubic//',num2str(i),'.bmp'];
    imwrite(K2, str5);
    L2 = imresize(I2,[2160,3840], 'lanczos2'); %Lanczos-2核
    str6 = ['images//720P//lanczos2//',num2str(i),'.bmp']; 
    imwrite(L2, str6);
end
