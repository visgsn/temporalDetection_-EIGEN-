function [laMiss,roc,gt,dt] = kaistTest( varargin )
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
dfs={ 'name','REQ', 'gtDir','REQ', 'bbsNm','REQ', 'pLoad',[], ...
  'thr',.5,'mul',0, 'ref',10.^(-2:.25:0), ...
  'lims',[3.1e-3 1e1 .05 1], 'show',1, 'type', '', 'clr', 'g', 'lineSt', '-' };
[name,gtDir,bbsNm,pLoad,thr,mul,ref,lims,show,type,clr,lineSt] = ...
  getPrmDflt(varargin,dfs,1);

% run evaluation using bbGt
[gt,dt] = bbGt('loadAll',gtDir,bbsNm,pLoad);
[gt,dt] = bbGt('evalRes',gt,dt,thr,mul);
[fp,tp,score,detR] = bbGt('compRoc',gt,dt,1,ref);  % detR = Detection Rate
laMiss=exp(mean(log(max(1e-10,1-detR))));  % miss=1-detR, laMiss: log-average
roc=[score fp tp];

% optionally plot roc
if( ~show ), return; end
figure(show); plotRoc([fp tp],'logx',1,'logy',1,'xLbl','False positives per image',...
  'lims',lims,'color',clr,'lineSt', lineSt,'smooth',1,'fpTarget',ref);
        

title(sprintf('log-average miss rate = %.2f%%',laMiss*100));
savefig([name type 'Roc'],show,'png');

end
