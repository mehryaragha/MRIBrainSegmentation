function demo

% This file is a demo for the MRI brain segmentation algorithm proposed in 
% M. Emambakhsh and M. Sedaaghi, “Automatic MRI brain segmentation using 
% local features, self-organizing maps, and watershed,” in IEEE 
% International Conference on Signal and Image Processing Applications 
% (ICSIPA 2009), Kuala Lumpur, Malaysia, 2009, pp. 129–134. 

% First feature extraction is performed using median, mean and
% morphological filtering. Dimensionality reduction and feature extraction
% are then performed by PCA and Self-Organizing Map Neural Networks
% (SOM-NN).
% Then the output is high-pass filtered to detect edges, which is then fed
% into a watershed segmentation.

% The code was written by Mehryar Emambakhsh
% Date: 24.09.2017
% Email: mehryar_emam@yahoo.com
% Paper:
% M. Emambakhsh and M. Sedaaghi, “Automatic MRI brain segmentation using 
% local features, self-organizing maps, and watershed,” in IEEE 
% International Conference on Signal and Image Processing Applications 
% (ICSIPA 2009), Kuala Lumpur, Malaysia, 2009, pp. 129–134. 

clc
close all

%%%%%% Read the image and pre-processing the image
% im = imread('test_image.TIF');
im = imread('test_image2.jpg'); im = rgb2gray(im);
figure(1), subplot(1, 2, 1), imshow(im), title('Input image')

% Isolating the brain region
im_b = im >  10;
im_b_filled = imfill(im_b, 'holes');
stats = regionprops(im_b_filled, 'area', 'centroid');
[~, max_ind] = max([stats.Area]);
all_cents = {stats.Centroid}; brain_cent = round(all_cents{max_ind});
L = bwlabel(im_b_filled);
cropping_map = L == L(brain_cent(1), brain_cent(2));
subplot(1, 2, 2), imshow(cropping_map), title('Brain region')
%%%%%%%%%%%%%%%%%%%%%

%%%%%%% Feature extraction
im = double(im);
im(~cropping_map) = nan;
toDisplay = 1;
feat_vector = feature_extraction(im, toDisplay);
%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% Dimensionality redution and clustering
[~, SCORE, ~] = princomp(feat_vector);
princ_comp = 11;
dim_red_feat_vector = SCORE(:, 1: princ_comp);

net = selforgmap([3 3], 100, 3, 'gridtop', 'linkdist');
net = train(net, dim_red_feat_vector');
y = net(dim_red_feat_vector');
cluster1_index = vec2ind(y);
myhist = hist(cluster1_index, 9);
feat_map_1 = reshape(myhist, [3, 3]);
feat_map_1 = feat_map_1/ sum(feat_map_1(:));

figure(5), 
imagesc(feat_map_1), title('9 X 9 SOM feature map'), 
colormap(gray)

net = selforgmap([4 1], 100, 3, 'gridtop', 'linkdist');
net = train(net, y);

output_feature_map = net(y);
cluster_index = vec2ind(output_feature_map);

clustered_map = im;
isNotNaN_index = find(~isnan(clustered_map));
for ind_cnt = 1: length(isNotNaN_index)
    clustered_map(isNotNaN_index(ind_cnt)) = cluster_index(ind_cnt);
end
figure(6),
imshow(clustered_map, []), 
figure(6), title('SOM clustering result')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Edge detection and watershed segmentation
clustered_map(isnan(clustered_map)) = 0;
clustered_map = medfilt2(clustered_map, [7, 7]);
im_edge_ver = imfilter(clustered_map, fspecial('sobel'));
im_edge_hor = imfilter(clustered_map, fspecial('sobel')');
im_edge = sqrt(im_edge_hor.^2 + im_edge_ver.^2);
figure(7), subplot(1, 3, 1), imshow(im_edge, []), title('Edge map')
im_seg = watershed(double(im_edge));

im_final = im;
im_final(im_seg == 0) = 255;
subplot(1, 3, 2), imshow(im_final, []), title('Segmentation map')
subplot(1, 3, 3), imshow(label2rgb(im_seg)), title('Segmentation colormap')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


