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

%% get parameters
dfs={ 'name','REQ', 'gtDir','REQ', 'bbsNm','REQ', 'pLoad',[], ...
  'thr',.5,'mul',0, 'ref',10.^(-2:.25:0), ...
  'lims',[3.1e-3 1e1 .05 1], 'show',1, 'type', '', 'clr', 'g', 'lineSt', '-' };
[name,gtDir,bbsNm,pLoad,thr,mul,ref,lims,show,type,clr,lineSt] = ...
  getPrmDflt(varargin,dfs,1);

%% run evaluation using bbGt
[gt,dt] = bbGt('loadAll',gtDir,bbsNm,pLoad);  % gt := gt0, dt := dt0
[gt,dt] = bbGt('evalRes',gt,dt,thr,mul);


%% ROC (Log-average miss rate)
[fp,tp,scoreROC,detR] = bbGt('compRoc',gt,dt,1,ref);  % detR = Detection Rate
laMiss=exp(mean(log(max(1e-10,1-detR))));  % miss=1-detR, laMiss: log-average
roc=[scoreROC fp tp];

% Plot results
f = figure(show);
plotRoc([fp tp],'logx',1,'logy',1,'xLbl','False positives per image',...
  'lims',lims,'color',clr,'lineSt', lineSt,'smooth',1,'fpTarget',ref,...
  'lineWd',2);
title(sprintf('log-average miss rate = %.2f%%',laMiss * 100));
% Save figure and image of figure
savefig([name type '_ROC' '.fig']);
frame=getframe(f);
[X,~]=frame2im(frame);
imwrite(X,[name type '_ROC' '.png'], 'png');
close(f);  % Close old figure

%% ROC (TP and FP)
% Plot results
f = figure(show);
f.Position(3) = f.Position(3) * 2;  % Double figure width
subplot(1,2,1);
plot(fp, tp, 'b', 'LineWidth', 2);
grid on;
title("true positives / false positives per image");
xlabel("FPPI [1 / image]"); ylabel("TP_{norm}");
% TP and FP over score
subplot(1,2,2);
plot(scoreROC, tp, 'g', scoreROC, fp, 'r', 'LineWidth', 2);
axis([0 100 0 min(3, max(max(fp), max(tp)))]);
grid on;
legend("TP_{norm}", "FPPI [1 / image]");
title("true / false positives w.r.t. score");
xlabel("score [%]"); %ylabel("[1 / image]");
% Save figure and image of figure
savefig([name type '_ROC-TP-FP' '.fig']);
frame=getframe(f);
[X,~]=frame2im(frame);
imwrite(X,[name type '_ROC-TP-FP' '.png'], 'png');
close(f);  % Close old figure

%% Precision and Recall
[rec,prec,scorePR,refPrec] = bbGt('compRoc',gt,dt,0,ref);
pr=[scorePR rec prec];

% Plot results
f = figure(show);
f.Position(3) = f.Position(3) * 2;  % Double figure width
subplot(1,2,1);
plot(rec * 100, prec * 100, 'b', 'LineWidth', 2);
axis([0 100 0 100]); grid on;
title("precision and recall");
xlabel("recall [%]"); ylabel("precision [%]");
% Precision and recall over score
subplot(1,2,2);
plot(scorePR, prec * 100, 'g', scorePR, rec * 100, 'b', 'LineWidth', 2);
axis([0 100 0 100]); grid on;
legend("precision", "recall");
title("precision / recall w.r.t. score");
xlabel("score [%]"); ylabel("[%]");
% Save figure and image of figure
savefig([name type '_PR' '.fig']);
frame=getframe(f);
[X,~]=frame2im(frame);
imwrite(X,[name type '_PR' '.png'], 'png');
close(f);  % Close old figure


%% Save workspace variables to file for further evaluation
save([name type '_WsVariables' '.mat']);


end
