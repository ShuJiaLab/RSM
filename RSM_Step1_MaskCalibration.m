
%% Parameters
r = 1.5; % radius of the digital pinhole
sz = 1024; % width of ROI
%sz = 2048;
maskSaveFileName = string(['INSER MASK NAME' datestr(now,'mmddyyyyHHMMSS')]);

%% Call ImageJ
ImageJ_call;

%% Load image

[file,path] = uigetfile('*.tif*');
if isequal(file,0)
    disp('User selected Cancel');
    return
else
    disp(['Loading ', fullfile(path,file)]);
    imp = ij.ImagePlus(fullfile(path,file));
    imp.show()
    im = MIJ.getCurrentImage;
end

%% Image pre-processing
im2 = double(im); 
ref0 = double(imgaussfilt(im2,50));
im2 = (im2./ref0)*100;
MIJ.createImage(int32(im2));
%% Find Maxima
MIJ.run("Find Maxima...", "prominence=10 output=List");
MX = MIJ.getResultsTable;
%% Create Mask
 % FociMap = zeros(1024,2048);
 % Mask= zeros(1024,2048); 
FociMap = zeros(1221,1024);
Mask= zeros(1221,1024); 
radius = 10;

for i = 1:length(MX)
    
    clc;disp(['Processing status: ' num2str(round(i/length(MX)*100)) '%']);
    
    if max(MX(i,:)<[12, 12]) || max(MX(i,:)>[size(im,2)-12, size(im,1)-12])
        continue
    end
    
    x_loc0 = MX(i,1) + 1;
    y_loc0 = MX(i,2) + 1;
    
    tmp = nrm(im2(y_loc0-radius:y_loc0+radius,x_loc0-radius:x_loc0+radius));
    
    s.StartPoint = [0,1,radius+1,radius+1,0,1];
    s.Lower = [0,0.9,0,0,0,3];
    [fitresult, gof] = Gauss2DFit(tmp,s,0);
    
    x_loc(i) = x_loc0 + (fitresult.c1-radius-1); %#ok<*SAGROW>
    y_loc(i) = y_loc0 + (fitresult.c2-radius-1);
    
    FociMap(round(y_loc(i)),round(x_loc(i))) = 1;
    
    x = 1:size(tmp,2);
    y = 1:size(tmp,1);
    [X, Y] = meshgrid(x,y);
    
    z = exp(-(((X-fitresult.c1)*cosd(fitresult.t1)+(Y-fitresult.c2)*sind(fitresult.t1))/r).^2-((-(X-...
        fitresult.c1)*sind(fitresult.t1)+(Y-fitresult.c2)*cosd(fitresult.t1))/r).^2);
    
    Mask(y_loc0-radius:y_loc0+radius,x_loc0-radius:x_loc0+radius) = z;
    
    
end

%% Save

save(maskSaveFileName,'Mask','FociMap','MX','x_loc','y_loc','-v7.3');

