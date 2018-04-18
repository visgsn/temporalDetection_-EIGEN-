function [miss,roc,gt,dt] = evalKAIST(outDir, dataDir, bbsNmFile)
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
%
% OUTPUTS
%  miss      - log-average miss rate computed at reference points
%  roc       - [nx3] n data points along roc of form [score fp tp]
%  gt        - [mx5] ground truth results [x y w h match] (see bbGt>evalRes)
%  dt        - [nx6] detect results [x y w h score match] (see bbGt>evalRes)


%% extract training and testing images and ground truth
cd(fileparts(which('evalKAIST.m')));
addpath( genpath( '..' ) );

%cond = { '-all', '-day', '-night' };
%for c=1:3,                                                                 % COMMENT OUT (WHOLE loop)!!!
%    type=['test' cond{c}]; skip=[];
%    dbInfo2(['kaist-' type]);
%    if(exist([dataDir type '/annotations'],'dir')), continue; end
%    dbExtract2(dataDir, type,1,skip);
%end
% Not used with caffe detector:
%type='train-all'; skip=20; 
%dbInfo2( ['kaist-' type] ); type=[ type int2str2(skip,2)];                 % COMMENT OUT (Return values needed?)???
%if(~exist([dataDir type '/annotations'],'dir')),  dbExtract2(dataDir, type,1,skip); end     % COMMENT OUT!!!

%% set up opts for training detector (see acfTrain) (ONLY DUMMY VARIABLES!)
opts=acfTrain2(); opts.modelDs=[50 20.5]; opts.modelDsPad=[64 32];
opts.pPyramid.smooth=.5;
opts.pPyramid.pChns.pColor.smooth=0; opts.nWeak=[32 128 512 2048];
opts.pBoost.pTree.maxDepth=2; 
opts.pBoost.pTree.fracFtrs=1/16; 
opts.pPyramid.pChns.pGradHist.softBin=1;
opts.pJitter=struct('flip',1);
opts.posGtDir=[dataDir 'train-all' int2str2(skip,2) '/annotations'];
opts.posImgDir=[dataDir 'train-all' int2str2(skip,2) '/images'];

%opts.name=[ 'models/AcfKAIST-RGB' ];
%opts.name=[ 'models/AcfKAIST-RGB-T' ];
%opts.name=[ 'models/AcfKAIST-RGB-T-TM-TO' ];
%opts.name=[ 'models/AcfKAIST-RGB-T-THOG' ];
opts.name=[ outDir 'results' ];                               % Output path

% Detection labels to use or ignore
pLoad={'lbls',{'person'},'ilbls',{'people','person?','cyclist'}};           % Adapt format=1 (PascalVoc)?!?
%pLoad={'format',1,'lbls',{'person'},'ilbls',{'people','person?','cyclist'}};% -->Modified version
opts.pLoad = [pLoad 'hRng',[55 inf], 'vType', {'none'} ];

%% To handle thermal channel
opts.imreadf = @imreadHistEq;

pCustom(1).enabled = 1;         % T          (T1)
pCustom(2).enabled = 0;         % TM+TO  (T2)
pCustom(3).enabled = 1;         % THOG    (T3)

pCustom(1).hFunc = @TRaw;
pCustom(2).hFunc = @TMagTOri;
pCustom(3).hFunc = @THog;

pCustom(1).name = 'T';
pCustom(2).name = 'TM+TO';
pCustom(3).name = 'THOG';

pCustom(1).pFunc = {};
pCustom(2).pFunc = {};
pCustom(3).pFunc = {};

opts.pPyramid.pChns.pCustom = pCustom;

%% train detector (see acfTrain)
% detector = acfTrain2( opts );                                               % COMMENT OUT!!!

%% modify detector (see acfModify)
pModify=struct('cascThr',-1,'cascCal',.025);                                % Doesn't matter if reapply == 0!!!
%detector=acfModify(detector,pModify);                                       % COMMENT OUT!!!

%% run detector on a sample image (only example) (see acfDetect)
cond = '-all';                                         % test subset to use
%cond = '-day';
%cond = '-night';

imgDir = [dataDir 'test' cond '/images'];
gtDir = [dataDir 'test' cond '/annotations'];

imgNms=bbGt('getFiles', {imgDir}, 2253, 4504);                              % Use f0 and f1 to select RGB-/IR-Images? (Or presort...)
I=imread(imgNms{102});
% Reimplement this with own detections!!!
%tic, bbs=acfDetect(I,detector); toc                                        % Give bbs with python! --> remove acfDetect(...)
%figure(3); imshow(I(:,:,1:3)); bbApply('draw',bbs); pause(.1);

%% test detector and plot roc (see acfTest)
[miss,roc,gt,dt]= kaistTest(...
  'name',opts.name,'imgDir',imgDir,...
  'gtDir',gtDir,'bbsNm',bbsNmFile,'pLoad',[pLoad, 'hRng',[55 inf],...
  'vType',{{'none','partial'}},'xRng',[5 635],'yRng',[5 475]],...
  'pModify',pModify,'reapply',0,'show',11,...
  'lims', [3.1e-3 3.1e1 .2 1],'type', cond, 'clr', 'r', 'lineSt', '-');
end