addpath(genpath("IALM-MC"));


data_folder = "../data/women_jpg";
num_images = 7;
for i = 1:num_images
    img = imread(fullfile(data_folder, sprintf("input_%d.jpg", i)));
    img = double(rgb2gray(img));
    if i == 1   
        h = size(img, 1);
        w = size(img, 2);
        data = double(zeros(num_images, h * w));
    end
    data(i, :, :) = reshape(img, 1, h * w);
end

[A, iter, svp] = inexact_alm_mc(data, 1e-7, 100, 1.0);
A = A.U * A.V.'; % recover a from SVD
A = reshape(A, num_images, h, w);
save("../data/women_jpg/A.mat", "A");
