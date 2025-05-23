clear,clc
close all
%% Recovery of ROI Shift with Cross-Correlation
% Shift a ROI by a known amount and recover the shift using cross-correlation.
sigma = 15; 
startFrame = 1;
frameNum = 350; % Total number of frames
w = 5;% additional width of the Search window
saveFileName = string(['ENTER FILE NAME' datestr(now,'mmddyyyyHHMMSS')]);
FineTracking_on =0;
%saveRefineTrack = string(['refineTracking_RSM_'
%datestr(now,'mmddyyyyHHMMSS')]); %if fine tracking is on


%% Step 1. Read Frames from File
% Load image file. This may be either TIFF or MAT
% Loads raw data and saves it under im
def_folder = 'enter data path';
[file,path] = uigetfile(def_folder,'*.mat'); 
[filepath,name,ext] = fileparts(file); %determines the extension of the chosen file
if isequal(file,0) 
    disp('User selected Cancel');
    return
elseif [ext] == '.mat'
    imb = load(fullfile(path, file));   

elseif [ext] == '.tif'
    imb = ReadTiff(fullfile(path, file));
    disp(['User selected ', fullfile(path,file)]);
end
%%
load('Mask from step 1', 'Mask', 'FociMap'); %load Mask

%% Step 2. Prepare frames for Analysis
% Smooth images to minimize the effect of point scanning
imgA = imgaussfilt(im(:,:,1),sigma);
% Normalize intensity
imgA = nrm(imgA); % Read first frame into imgA
% final image imgA is normalized and gaussian fitted im

%% Step 3. Select Sample Window
%displays imgA with scaled colors - full raw image
imagesc(imgA); colormap hot; axis image
h = drawrectangle;
%coordinates of the selected region (ROI)
%coords1=x top left corner, coords2 = y top left corner coords3 = width,
%coords 4 = height
coords = round(h.Position);
close;
%Window selected from imgA within the coordinates
Sect = imgA(coords(2):coords(2)+coords(4)-1,coords(1):coords(1)+coords(3)-1);
% bim(Sect)

% Inizialization - Helps save memory for big data for-loop

%frameNum = The minimum value of (framenum, the length of the third dim in
%im(500~#frames), - the first frame+1`2nd) == specifies the number of frame
%to analyze. What the user choose and not how many there are in the raw
%image
frameNum = min(frameNum,size(im,3)-startFrame+1);

% creates 3D array of zeros under S with size of x and y of Sect and z of
% frameNum
S = zeros(size(Sect,1),size(Sect,2),frameNum);

%Creates a focimap for S: zero matrix with size of S
S_focimap = zeros(size(S));

%Creates a 2 by frameNum matrix of zeros ~ Top left corner
Positions = zeros(2,frameNum);

% puts the coordinates of 2 and 1 in the first column of position
Positions(:,1) = [coords(2) coords(1)];
% Add selected window to the final output video
S(:,:,1) = nrm(im(coords(2):coords(2)+coords(4)-1,coords(1):coords(1)+coords(3)-1));

% Define width of Search window-where we are going to look for our image based on the coordinates: from up and down and a little 
%left and right defined by w 
SearchWidth = max(1,coords(1)-w):min(size(imgA,2),coords(1)+coords(3)+w-1);

imax0 = 0;

endFrame = frameNum+startFrame-1;
%%
%Step1 intensity variance
intensityVariance = zeros(1, frameNum);  % Pre-allocate array for intensity variance

for i = 1:frameNum
    currentFrame = im(:,:,i);  % Extract the i-th frame
    % Calculate variance of the current frame
    intensityVariance(i) = var(double(currentFrame(:)));
end

%%
%Step 2 calculate variance

highVarianceThreshold = mean(intensityVariance) + std(intensityVariance);
lowVarianceThreshold = mean(intensityVariance) - std(intensityVariance);

%% Cross corelation tracking method

record_xy=zeros(frameNum,2); %record tracking value
%Goes frame by frame and tries to find the corresponding window
for i = startFrame:endFrame-1
% for i = 1:270
    clc
    disp(['Processing frame ' num2str(i+1) '/' num2str(endFrame+1)])
    currentVariance = intensityVariance(i);  % Assuming intensityVariance is calculated beforehand
    
    % Adjust tracking parameters based on intensity variance
    if currentVariance > highVarianceThreshold
        % High variance adjustments
        adjustedSigma = sigma * 1.5;  % Example: Increase Gaussian filter sigma
        % Possibly adjust Sect size here
    elseif currentVariance < lowVarianceThreshold
        % Low variance adjustments
        adjustedSigma = sigma * 0.75;  % Example: Decrease Gaussian filter sigma
        % Possibly adjust Sect size here
    else
        % Default parameters for normal variance
        adjustedSigma = sigma;
    end
    
    %img0 = image of raw data first position +1 `i = frameNum
    img0 = im(:,:,i+1);
    % We are only going to look within the Search window and not the whole
    img = img0(:,SearchWidth); %img = original image
    imgB = imgaussfilt(img,sigma);
    % Normalize intensity
    imgB = nrm(imgB);

    %%  Step 4. Cross-correlation
    % Cross-correlate the two matrices and find the maximum absolute value of
    % the cross-correlation. Use the position of the maximum absolute value to determine
    % the shift in the template.
    
    
    % finds any cross relation between imgB and Sect~the selected window
    %To find imgB in the Section-from the first frame
    %moves section across imgB and measure cc for each number
    %imgB-biggerimage Sect-small img
    cc = xcorr2(imgB,Sect);
    [max_cc, imax] = max(abs(cc(:)));
 
    %Find peaks of the correlation
    %max_cc=value
    %imax = location
    [max_cc, imax] = findpeaks(abs(cc(:)),'SortStr','descend','MinPeakDistance',2.5e5);
    
    %evt is being saved in imax0
    if imax0 == 0
        imax0 = imax(1); %if the first imax~peak location is 0 then it is equal to imax(1) which is 2.5e5 
    elseif imax(1)-imax0 > 2.5e5 %if peak separation - loc of first peak is >2.5e5 then imax0 = 5e5 ~2nd peak position
        imax0 = imax(2);
    else
        imax0 = imax(1); %else loc of peak 1 = imax(1)
    end
    
    [ypeak, xpeak] = ind2sub(size(cc),imax0);

    xpeak = max(xpeak,size(Sect,2)); 
    ypeak = max(ypeak-1,size(Sect,1));
    
    %coordinate of top left corner in the new frame [coordinates 1 and 2 of
    %the second frame and so on] just in the search area
    %saves the peaks under c_offset. subtracts size(sect) to get the top left position xpeak B-h = T ypeakr-w=L
    %coffset is the yposition
    c_offset = [(ypeak-size(Sect,1)+1) (xpeak-size(Sect,2)+1)]; 
    record_xy(i,:)=c_offset;

    % figure
    % plot(cc(:))
    % title('Cross-Correlation')
    % hold on
    % plot(imax,max_cc,'or')
    % hold off
    % text(imax*1.05,max_cc,'Maximum')
    % disp(corr_offset)
    
    %% Step 5. Use correlation offset to cut the new frame
    % The shift obtained from the cross-correlation equals the known template
    % shift in the row and column dimensions.
    
    j = i-startFrame+2;
    
   
    S(:,:,j) = nrm(rot90(img(ypeak:-1:c_offset(1),xpeak:-1:c_offset(2)),2));

    %Positions of the frames in ref to the original coordinates
    %Can now use this position to do the tracking in a different z without
    %having to remember the area that we selected
    Positions(:,j) = [c_offset(1), c_offset(2)+SearchWidth(1)-1];
    
    %foci position doesn't change with z
   S_focimap(:,:,j) = FociMap(Positions(1,j):Positions(1,j)+coords(4)-1,...
        Positions(2,j):Positions(2,j)+coords(3)-1);
     
     % S_focimap(:,:,j) = shiftedFoci(Positions(1,j):Positions(1,j)+coords(4)-1,...
     %     Positions(2,j):Positions(2,j)+coords(3)-1);
    
    % imagesc(Sect)
    % axis image off
    % colormap gray
    % title('Reconstructed')
    % figure; imshowpair(Sect,Sect,'ColorChannels','red-cyan');
    % title('Color composite (frame A = red, frame B = cyan)');
    
end



%% Step 6. Show & Save
figure; imshowpair(S(:,:,1),S(:,:,end),'ColorChannels','red-cyan');
title('Color composite (frame A = red, frame B = cyan)');
implay(S);
save(saveFileName,'S','S_focimap','coords','Positions','-v7.3')


%% Step 7. Refine tracking

record_xyR=zeros(frameNum,2);
if FineTracking_on
    frameNum = size(S,3);
    sigma = 12;
    % Normalize intensity
    imgA = nrm(imgaussfilt(S(:,:,2),sigma)./imgaussfilt(S(:,:,2),500)); % Read first frame into imgA
    % Select Sample Window
    imagesc(imgA)
    daspect([1 1 1])
    h = drawrectangle;
    coords2 = round(h.Position);
    close;
    Sect2 = imgA(coords2(2):coords2(2)+coords2(4)-1,coords2(1):coords2(1)+coords2(3)-1);
    % Inizialization
    corr_data = zeros(size(Sect2,1),size(Sect2,2),frameNum);
    corr_focimap = zeros(size(corr_data));
    Positions2 = zeros(2,frameNum);
    Positions2(:,1) = Positions(:,1) + [coords2(2) coords2(1)]';
    % Add selected window to the final output video
    corr_data(:,:,1) = (S(coords2(2):coords2(2)+coords2(4)-1,coords2(1):coords2(1)+coords2(3)-1));
    
    for i = 2:frameNum
        clc
        disp(['Fine correlation (Processing frame ' num2str(i) '/' num2str(frameNum) ')'])
        imgB = nrm(imgaussfilt(S(:,:,i),8)./imgaussfilt(S(:,:,i),50)); % Read new frame into imgB
        cc = xcorr2(imgB,Sect2);
        [max_cc, imax] = max(abs(cc(:)));
        [ypeak, xpeak] = ind2sub(size(cc),imax(1));
        c_offset = [abs(ypeak-size(Sect2,1)+1) abs(xpeak-size(Sect2,2)+1)];
        corr_data(:,:,i) =rot90(S(ypeak:-1:c_offset(1),xpeak:-1:c_offset(2),i),2);
        corr_focimap(:,:,i) = rot90(S_focimap(ypeak:-1:c_offset(1),xpeak:-1:c_offset(2),i),2);
        Positions2(:,i) = Positions(:,i) + [c_offset(1)-1, c_offset(2)-1]';
    end
    
    figure; imshowpair(corr_data(:,:,1),corr_data(:,:,end),'ColorChannels','red-cyan');
    title('Color composite (frame A = red, frame B = cyan)');
    implay(corr_data);
    
    
end
%%
save(saveRefineTrack,'corr_data','corr_focimap','Positions2','-v7.3')
save(saveFileName,'S','S_focimap','Positions','-append')

%% 
% 1D tracking method
S_1D=0*S;
S_foci1D=0*S;

yfitf = record_xy(:,2);
 
for j=1:350
    j;
    img0 = im(:,:,j);
    img = img0(:,SearchWidth);
    foci = FociMap(:,SearchWidth);
%shift foci parameters incase raw image and focimap are shifted
shiftLeft =0;   % Adjust as needed
shiftDownwards =0;   % Adjust as needed

% Shift the foci map matrix
shiftedFoci = circshift(foci, [shiftDownwards, shiftLeft, 0]);

      S_1D(:,:,j) = (rot90((img(yfitf(j)+coords(4):-1:yfitf(j)+1,end-w-1:-1:w)),2));
      S_foci1D(:,:,j) = rot90(shiftedFoci(yfitf(j)+coords(4):-1:yfitf(j)+1,end-w-1:-1:w),2);
end

figure; imshowpair(S_1D(:,:,1),S_1D(:,:,end-1),'ColorChannels','red-cyan');
title('Color composite (frame A = red, frame B = cyan)');
save(saveFileName,'S_1D','S_foci1D','coords','X2','-v7.3')
