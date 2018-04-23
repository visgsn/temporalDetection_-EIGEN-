function [hs, hImg] = kaistShow( varargin )
% Test aggregate channel features object detector given ground truth.
%
% USAGE
%  [miss,roc,gt,dt] = kaistTest( pTest )
%
% INPUTS
%  pTest    - parameters (struct or name/value pairs)
%   .name     - ['REQ'] detector name
%   .gtDir    - ['REQ'] dir containing test ground truth
%   .bbsNm    - ['REQ'] file containing detections (.txt format)
%   .pLoad    - [] params for bbGt>bbLoad for test data (see bbGt>bbLoad)
%   .thr      - [.5] threshold on overlap area for comparing two bbs
%   .mul      - [0] if true allow multiple matches to each gt
%   .ref      - [10.^(-2:.25:0)] reference points (see bbGt>compRoc)
%   .lims     - [3.1e-3 1e1 .05 1] plot axis limits
%   .show     - [0] optional figure number for display
%
% OUTPUTS
%  miss     - log-average miss rate computed at reference points
%  roc      - [nx3] n data points along roc of form [score fp tp]
%  gt       - [mx5] ground truth results [x y w h match] (see bbGt>evalRes)
%  dt       - [nx6] detect results [x y w h score match] (see bbGt>evalRes)
%
% EXAMPLE
%
% See also acfTrain, acfDetect, acfModify, acfDemoInria, bbGt
%
% Piotr's Computer Vision Matlab Toolbox      Version 3.40
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

% get parameters
dfs={ 'name','REQ', 'gtDir','REQ', 'bbsNm','REQ', 'imgNms','REQ', ...
    'pLoad',[], 'thr',.5, 'mul',0, 'ref',10.^(-2:.25:0), ...
    'lims',[3.1e-3 1e1 .05 1], 'show',1, 'type','', 'clr','g', ...
    'lineSt','-' };
[name,gtDir,bbsNm,imgNms,pLoad,thr,mul,ref,lims,show,type,clr,lineSt] = ...
  getPrmDflt(varargin,dfs,1);

% Open figure and display buttons for user interaction (window control)
figure();
a = uicontrol('Style','pushbutton', 'Position',[20 220 60 40],...
              'String','ABORT', 'Callback',@forwardBackward);
f = uicontrol('Style','pushbutton', 'Position',[20 170 60 40],...
              'String','>', 'Callback',@forwardBackward);
b = uicontrol('Style','pushbutton', 'Position',[20 120 60 40],...
              'String','<', 'Callback',@forwardBackward);
ff = uicontrol('Style','pushbutton', 'Position',[20 70 60 40],...
               'String','>>', 'Callback',@forwardBackward);
fb = uicontrol('Style','pushbutton', 'Position',[20 20 60 40],...
               'String','<<', 'Callback',@forwardBackward);
sl = uicontrol('Style','slider', 'Position',[100 20 400 15],...
               'Min',1, 'Max',length(imgNms), 'Value',1,...
               'String','goto', 'Callback',@forwardBackward);

% run evaluation using bbGt
[gt,dt] = bbGt('loadAll',gtDir,bbsNm,pLoad);
[gt,dt] = bbGt('evalRes',gt,dt,thr,mul);
imgCount = 1;
while imgCount <= length(imgNms)
    img = imread(imgNms{imgCount});
    grtr = gt{imgCount};
    det = dt{imgCount};
    [hs,hImg] = bbGt( 'showRes', img, grtr, det);
    
    % print current array index and image name
    fprintf(...
        'Current image (iter, name): %d/%d, %s\n',...
        imgCount, length(imgNms), imgNms{imgCount});
    
    % Pause execution and wait for user input
    uiwait(gcf);
end


%% Function to manage user inputs
function forwardBackward(source, event)
    % Adapt values according to input
    if source.String == ">"
        imgCount = imgCount + 1;
    elseif source.String == "<"
        imgCount = imgCount - 1;
    elseif source.String == ">>"
        imgCount = imgCount + 20;
    elseif source.String == "<<"
        imgCount = imgCount - 20;
    elseif source.String == "goto"
        imgCount = floor(source.Value);
    elseif source.String == "ABORT"
        error('Program aborted by user.')
    end
    % Keep value inside feasible borders
    if imgCount < 1
        imgCount = 1;
    end
    % Assign current value to slider
    sl.Value = imgCount;
    % Proceed execution
    uiresume(gcbf);
end

end
