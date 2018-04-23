function [miss,roc,gt,dt] = evalKAIST(inOutDir, dataDir, bbsNmFile, showRes)
% Demo for aggregate channel features object detector on KAIST dataset.
%
% See also acfReadme.m
%
% Piotr's Computer Vision Matlab Toolbox      Version 3.40
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]
%
% 2015.06.02. Modified by Soonmin Hwang [smhwang-at-rcv.kaist.ac.kr]
% 2015.07.27. Some bugs are fixed. 
%   - dbInfo2.m
%   - detector/acfDemoKAIST.m
%   - detector/acfTest.m
%   - channels/chnsPyramid.m
%   - channels/TMagTOri.m (added)
%
% INPUTS
%  outDir    - Directory to save test results (With trailing "/"!)
%  dataDir   - Directory containing KAIST data (DEFAULT: $KAIST_root/data-kaist/)
%  bbsNmFile - Detections (must be of form [x y w h wt bbType] as .txt file)
%  showRes   - 0: Perform testing  1: Show test results (Previous testing required!)
%
% OUTPUTS
%  miss      - log-average miss rate computed at reference points
%  roc       - [nx3] n data points along roc of form [score fp tp]
%  gt        - [mx5] ground truth results [x y w h match] (see bbGt>evalRes)
%  dt        - [nx6] detect results [x y w h score match] (see bbGt>evalRes)


%% Initialize basic variables and paths
tmpPath = fileparts(which('evalKAIST.m'));
addpath(genpath(sprintf(tmpPath, '/..')));
[~,bbsNmName,~] = fileparts(bbsNmFile);

if ~exist(bbsNmFile, 'file')
    error('evalKAIST:DetectionFileDoesNotExist', ...
        'Detection file could not be found!');
end

%% set up opts for detector (see acfTrain - ONLY DUMMY VARIABLES!)
opts=acfTrain2();

%% ##### CONFIGURATION OPTIONS ############################################
opts.name=[ inOutDir bbsNmName ];                     % Input and Output path

% Detection labels to use or ignore
pLoad={'lbls',{'person'},'ilbls',{'people','person?','cyclist'}};

% Specify test subset to use
cond = '-all';
%  ########################################################################

imgDir = [dataDir 'test' cond '/images'];
gtDir = [dataDir 'test' cond '/annotations'];




%% Test detector or show detection results
% Show results
if showRes
    % Use f0 and f1 to select RGB-/IR-Images (RGB: 1, 2252 / IR: 2253, 4504)
    imgNms=bbGt('getFiles', {imgDir}, 2253, 4504);
    
    [hs, hImg]= kaistShow(...
      'name',opts.name,...
      'gtDir',gtDir,...
      'bbsNm',bbsNmFile,...
      'imgNms',imgNms,...
      'pLoad',[pLoad, 'hRng',[55 inf],...
      'vType',{{'none','partial'}},'xRng',[5 635],'yRng',[5 475]],...
      'show',11,...
      'lims',[3.1e-3 3.1e1 .2 1],...
      'type',cond,...
      'clr','r',...
      'lineSt','-');
    
    %I=imread(imgNms{102});
    % Reimplement this with own detections!!!
    %tic, bbs=acfDetect(I,detector); toc
    %figure(3); imshow(I(:,:,1:3)); bbApply('draw',bbs); pause(.1);
    
    
% Do testing
else
    %% test detector and plot roc (see acfTest)
    [miss,roc,gt,dt]= kaistTest(...
      'name',opts.name,...
      'gtDir',gtDir,...
      'bbsNm',bbsNmFile,...
      'pLoad',[pLoad, 'hRng',[55 inf],...
      'vType',{{'none','partial'}},'xRng',[5 635],'yRng',[5 475]],...
      'show',11,...
      'lims',[3.1e-3 3.1e1 .2 1],...
      'type',cond,...
      'clr','r',...
      'lineSt','-');
end


end
