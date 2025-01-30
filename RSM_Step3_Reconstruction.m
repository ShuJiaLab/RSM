%% scanOSM reconstruction

close all;

%% Parameters
tic;
% Load data from file
Load_file =0;
% Save data on the same file
Save_on =1;
% Perform deconvolution
Decon_on =1;
% Starting frame
sFrame = 1;
% Number of frames to analyze
nFrames =350;
% Perform 2D Gaussian fit (only for beads)
Gaussfit =0;
% Display resolution for each fit
Verbose = 0;
% Radius digital pinhole
r =1;

save_tiff = 1;

alpha = 1.25;
beta = 1.1; 


%% Load Data

if Load_file
    
    def_folder = '';
    
    [file,path] = uigetfile(def_folder,'*.mat');
    if isequal(file,0)
        disp('U ser selected Cancel');
        return    
    else
         disp(['Loading ', fullfile(path,file)]);
         load(fullfile(path,file));
        
      
    end
end
%
%% 
%load tracked data  
cdata0 = S_1D;
data_focimap = S_foci1D;
  
%% Bkg removal
disp('Background removal...')

nFrames = min(nFrames,size(cdata0,3)-sFrame+1);

% creates a matrix of zeros based on the sizes of corr_data's row and col[all the tracked frames]
% and nFrames
cdata = zeros(size(cdata0,1),size(cdata0,2),nFrames);

se = strel('disk',10);

modulation = squeeze(mean(cdata0,[1,2])).\mean2(cdata0(:,:,1));

sigma_gauss =150;
 for idx = 1:nFrames
   
     i =idx+sFrame-1;
    
%multiple methods to remove background
        % cdata(:,:,idx) = (imsubtract(imadd(cdata0(:,:,i),imtophat(cdata0(:,:,i),se)),imbothat(S(:,:,i),se)));
      %cdata(:,:,idx) = modulation(i).*(imsubtract(imadd(cdata0(:,:,i),imtophat(cdata0(:,:,i),se)),imbothat(cdata0(:,:,i),se)));
         cdata(:,:,idx) = (cdata0(:,:,i)./imgaussfilt(cdata0(:,:,i),sigma_gauss));
          %cdata(:,:,idx) = (cdata0(:,:,i)./imgaussfilt(cdata0(:,:,i),sigma_gauss));
          %cdata(:,:,idx) = modulation(i).*(cdata0(:,:,i)./imgaussfilt(cdata0(:,:,i),sigma_gauss));
    % cdata(:,:,idx) = modulation(i).^alpha.*(corr_data(:,:,i));
     
     %         widefield0(:,:,idx) = modulation(i).*S(:,:,i+startFrame-1);
 end

%%
 widefield = (imresize(sum(cdata(:,:,sFrame:sFrame+nFrames-1),3),2,'nearest'));
%% Pixel Reassignment
disp('Pixel Reassignment...')

clear output*

%pads the frame of the bkg removed image with 0's. 

Q = padarray(((cdata)),[r r],0,'both'); 
Q = Q + 0.0001;

%Generates an output image twice the size of Q-padded data of our image
output = zeros(2*size(Q,1),2*size(Q,2)); 

% overlap detector
overlap=output;
overlap = padarray((overlap),[r r],0,'both'); 
overlap=logical(overlap);
checker=overlap;
output_w = ones(2*size(Q,1),2*size(Q,2));

%size of our image
sz2 = size((nrm(cdata(:,:,1))));
% spot_visited = zeros(sz2);



for i = 1:nFrames
    clear k
    
    %find elements that are greater than 0.95 in the focimap
    k = find((data_focimap(:,:,i))>0.95);
     
    %uses the focimap to place each pixel in the new bigger image
    for j = 1:length(k)
        
        %changes the linear index of cdata based on the foci map and
        %assisgns it to row and column
        [row,col] = ind2sub(sz2,k(j));
        %                         if spot_visited(row,col) == 1
        %                             continue
        %                         else
        %                         spot_visited(row,col) = 1;
        %                         end
        row = row + r;
        col = col + r;
        for jj = -r:+r
            for ii = -r:+r
                
                if overlap(1+2*(row-1)+ii,1+2*(col-1)+jj)~=0
                       %output(1+2*(row-1)+ii,1+2*(col-1)+jj) = (output(1+2*(row-1)+ii,1+2*(col-1)+jj)+Q(row+ii,col+jj,i));
                    % output(1+2*(row-1)+ii,1+2*(col-1)+jj) = (output(1+2*(row-1)+ii,1+2*(col-1)+jj)+Q(row+ii,col+jj,i))/2;
                   output(1+2*(row-1)+ii,1+2*(col-1)+jj) = max(output(1+2*(row-1)+ii,1+2*(col-1)+jj), Q(row+ii,col+jj,i));

                    checker(1+2*(row-1)+ii,1+2*(col-1)+jj) = 1;
                
                else
                    %output(1+2*(row-1)+ii,1+2*(col-1)+jj) =  (output(1+2*(row-1)+ii,1+2*(col-1)+jj) + Q(row+ii,col+jj,i));
                     output(1+2*(row-1)+ii,1+2*(col-1)+jj) = (max(output(1+2*(row-1)+ii,1+2*(col-1)+jj), Q(row+ii,col+jj,i)));
                      checker(1+2*(row-1)+ii,1+2*(col-1)+jj) = 1;
                     output_w(1+2*(row-1)+ii,1+2*(col-1)+jj) = output_w(1+2*(row-1)+ii,1+2*(col-1)+jj) + 1;
                end
                overlap(1+2*(row-1)+ii,1+2*(col-1)+jj) = 1;
                
            end
        end
    end
end
% figure;
% imagesc(output(11:end-10,11:end-10)/2);
% axis image
% title Before filling
% colormap gray
%% weighted average - 
disp('Image Reconstruction...')

output = removePad(output,2*r);
output_w = removePad(output_w,2*r);

output = nrm(output./output_w.^beta);
%output = (output./output_w.^beta);



figure;
imagesc(output(11:end-10,11:end-10)/2);
axis image
title Before filling
colormap gray

outputz=output;
outputz(outputz==0)=NaN;

F = fillmissing(outputz,'spline',1,'MaxGap',10);


% save outputz outputz

figure;
imagesc(F(11:end-10,11:end-10)/2);
axis image
title stage 1
colormap gray

F1 = fillmissing(F,'spline',2,'MaxGap',10);

outputz=output;
outputz(outputz==0)=NaN;
F = fillmissing(outputz,'spline',2,'MaxGap',10);
F2 = fillmissing(F,'spline',1,'MaxGap',10);

F=F1;
F2D = inpaint_nans(outputz(11:end-10,11:end-10),4);
%F2D = imgaussfilt(F2D,0.75);

figure;
imagesc(F2D(11:end-10,11:end-10)/2);
axis image
title 2D inpaint
colormap gray

%output = imgaussfilt(F,0.75);

figure;
imagesc(widefield);
axis image
title Widefield
colormap gray

output(output>1)=1;

figure;
imagesc(nrm(output(11:end-10,11:end-10)/2));
axis image
title INT
colormap gray



%% Deconvolution
if Decon_on
    disp('Deconvolution...')
    
    %lambda =515; NA=1.45;.61*lambda/NA/sqrt(2)
    sigma_PSF = 153.198/2.345/32.5; % 488 laser
    %sigma_PSF = 200.4/2.335/32.5; %647 laser
    PSF = fspecial('gaussian',2*ceil(2*sigma_PSF)+1,sigma_PSF);
    [usim,PSF] = deconvblind(output,PSF,5); % suggested # of iterations: 5-8
    [usim2,PSF] = deconvblind(F2D,PSF,5); % suggested # of iterations: 5-8
   
  
    figure;
    imagesc(nrm(usim));
    axis image
    title ('RSM deconv')
    colormap gray
end
%%

%dECONVOLUTION IF THERE ARE NEGATIVE VALUES
% Check for NaN or Inf values and their indices
nan_inf_indices = isnan(output) | isinf(output);
num_nan_inf = sum(nan_inf_indices(:));
disp(['Number of NaN or Inf values: ', num2str(num_nan_inf)]);

% Handle NaN or Inf values by replacing them with zero
%output(nan_inf_indices) = 0;

% Alternatively, handle NaN or Inf values by replacing them with the mean of the surrounding values
 output(nan_inf_indices) = mean(output(~nan_inf_indices));

% Check for NaN or Inf values again after handling
if any(isnan(output(:))) || any(isinf(output(:)))
    error('The input image still contains NaN or Inf values after handling.');
end

% Ensure the data types are correct
if ~isa(output, 'double') || ~isa(PSF, 'double')
    error('Both the input image and PSF must be of type double.');
end

% Check dimensions
[output_rows, output_cols] = size(output);
[psf_rows, psf_cols] = size(PSF);

if output_rows < psf_rows || output_cols < psf_cols
    error('PSF dimensions are larger than the output image dimensions.');
end

% Normalize the PSF
PSF = PSF / sum(PSF(:));

% Display value ranges
disp(['Range of output: ', num2str(min(output(:))), ' to ', num2str(max(output(:)))]);
disp(['Range of PSF: ', num2str(min(PSF(:))), ' to ', num2str(max(PSF(:)))]);

% Visualization
figure;
subplot(1,2,1);
imshow(output, []);
title('Output Image');

subplot(1,2,2);
imshow(PSF, []);
title('PSF');

% Perform deconvolution with a simpler PSF as a debugging step
[usim, PSF] = deconvblind(output, PSF, 5); % Suggested number of iterations: 5-8
[usim2,PSF] = deconvblind(F2D,PSF,5); % suggested # of iterations: 5-8
%[usim3,PSF] = deconvblind(intpaF,PSF,5); % suggested # of iterations: 5-8
figure;
    imagesc(nrm(usim));
    axis image
    title ('RSM DECONV')
    colormap gray


    figure;
    imagesc(nrm(usim2));
    axis image
    title ('RSM DECONV2')
    colormap gray
%% Save results
if Save_on
    disp('Saving...')
    def_folder = 'ENTER PATH';
    
    path = def_folder;
    save(fullfile(path,'DATA_NAME'),'S_1D','S_foci1D');
end

disp('Done.')
toc;

%%
def_folder = 'ENTER PATH';
    
    path = def_folder;
if save_tiff
    saveastiff(uint16((F2D(11:end-10,11:end-10)).*2^16),fullfile(path,'INT2.tif '));
    saveastiff(uint16(nrm(widefield(11:end-10,11:end-10)).*2^16),fullfile(path,'WF.tif'));
    saveastiff(uint16((output(11:end-10,11:end-10)).*2^16),fullfile(path,'INT.tif '));
    saveastiff(uint16((usim).*2^16),fullfile(path,'RSM.tif'));  
    saveastiff(uint16((usim2).*2^16),fullfile(path,'RSM2.tif')); 
     
end

disp('Done.')
