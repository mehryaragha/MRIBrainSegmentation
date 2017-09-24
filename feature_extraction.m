function feat_vector = feature_extraction(im, toDisplay)
% This function performs feature extraction over the input image and the
% result would be in the form of equation (4) in the paper.

% Inputs: im is an I X J image
% Outputs: feat_vector is an N X d matrix, which N is the number of samples
% and d is the feature vector dimensionality.


% The code was written by Mehryar Emambakhsh
% Date: 24.09.2017
% Email: mehryar_emam@yahoo.com
% Paper:
% M. Emambakhsh and M. Sedaaghi, “Automatic MRI brain segmentation using 
% local features, self-organizing maps, and watershed,” in IEEE 
% International Conference on Signal and Image Processing Applications 
% (ICSIPA 2009), Kuala Lumpur, Malaysia, 2009, pp. 129–134. 

feat_vector = im(~isnan(im));

all_mask_sizes = 3: 2: 9;
%%%%%%%%% Mean features
if toDisplay
    figure(2), title('Mean features')
end
for mask_cnt = 1: length(all_mask_sizes)
    curr_mask = all_mask_sizes(mask_cnt);
    medFun = @(x) nansum(nansum(fspecial('gaussian', [curr_mask, curr_mask], 3).* x));
    im_filtered = nlfilter(im, [curr_mask curr_mask], medFun);
    feat_vector = [feat_vector, im_filtered(~isnan(im))];
    if toDisplay
        subplot(1, length(all_mask_sizes), mask_cnt)
        imshow(im_filtered, [])
        title(['Gaussian ' num2str(mask_cnt)])
    end
end
%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%% Median features
if toDisplay
    figure(3), title('Median features')
end
for mask_cnt = 1: length(all_mask_sizes)
    curr_mask = all_mask_sizes(mask_cnt);
    medFun = @(x) nanmedian(x(:));
    im_filtered = nlfilter(im, [curr_mask curr_mask], medFun);
    feat_vector = [feat_vector, im_filtered(~isnan(im))];
    if toDisplay
        subplot(1, length(all_mask_sizes), mask_cnt)
        imshow(im_filtered, [])
        title(['Median ' num2str(mask_cnt)])
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%% Morphological features
all_lengths = [3, 7];
all_angles = 0: 60: 120;

im_new = im;
im_new(isnan(im)) = 0;

if toDisplay
    curr_ind = 1;
    figure(4), title('Morphological features')
end
for len_cnt = 1: length(all_lengths)
    for ang_cnt = 1: length(all_angles)
        curr_SE = strel('line', all_lengths(len_cnt), all_angles(ang_cnt));
        curr_morph = imopen(imclose(im_new, curr_SE), curr_SE);
        feat_vector = [feat_vector, curr_morph(~isnan(im))];
        
        if toDisplay
            subplot(2, length(all_angles), curr_ind)
            imshow(curr_morph, [])
            title(['Morphological' num2str(curr_ind)])
            curr_ind = curr_ind + 1;
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

