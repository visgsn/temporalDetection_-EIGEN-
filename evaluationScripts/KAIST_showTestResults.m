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
%  outDir    - Directory to save test results (With trailing '/'!)
%  dataDir   - Directory containing KAIST data (DEFAULT: $KAIST_root/data-kaist/)
%  bbsNmFile - Detections (must be of form [x y w h wt bbType] as .txt file)
%
% OUTPUTS
%  miss      - log-average miss rate computed at reference points
%  roc       - [nx3] n data points along roc of form [score fp tp]
%  gt        - [mx5] ground truth results [x y w h match] (see bbGt>evalRes)
%  dt        - [nx6] detect results [x y w h score match] (see bbGt>evalRes)


%% Set up necessary path variables
%##########################################################################
% Change "atWORK" to switch between HOME and WORK directories (False: HOME - True: WORK)
atWork = true;

iterToTest     = '50184';  % Specify which solver iteration Nr. to test

subsetName     = 'train-all-T';
job_name       = 'refinedet_it50184_320x320';  % DEFAULT: 'refinedet_vgg16_320x320'
experimentName = 'testing';  % Name of evalOutput subfolder (experiment name)

% Path prefix for output directory
path_prefix_HOME = sprintf('%s/train_test_data', getenv('HOME'));
path_prefix_WORK = '/net4/merkur/storage/deeplearning/users/gueste/TRAINING_test';
% Path to dataset root (e.g. '/home/gueste/data/KAIST/data-kaist/')
dataset_root_HOME = sprintf('%s/data/KAIST/data-kaist/', getenv('HOME'));
dataset_root_WORK = '/net4/merkur/storage/deeplearning/users/gueste/data/KAIST/data-kaist/';
%##########################################################################

%% Construct basic paths from config
if atWork == true
    path_prefix = path_prefix_WORK;
    dataset_root = dataset_root_WORK;
else
    path_prefix = path_prefix_HOME;
    dataset_root = dataset_root_HOME;
end

if ( ~exist(path_prefix, 'dir') || ~exist(dataset_root, 'dir') )
    error('KAIST_showTestResults:DirectoryNotFound', ...
        'Paths for path_prefix and/or dataset_root do not exist! --> atWORK = ?!');
end

train_test_outPath = sprintf(...
    '%s/models/VGGNet/KAIST/%s/%s',...
    path_prefix, subsetName, job_name);

%% Construct complete input paths
inOutDir = sprintf(...
    '%s/evalOutput/%s/KAIST_%s_iter_%s/detections_for_matlab/',...
    train_test_outPath, experimentName, job_name, iterToTest);
dataDir = dataset_root;
% Search for correct input file with person detections
filesInOutDir = dir(inOutDir);  % List all files in inOutDir
for i = 1:length(filesInOutDir)
    if contains(filesInOutDir(i).name, 'person.txt')
        bbsNmFile = sprintf(...
            '%s/%s', filesInOutDir(i).folder, filesInOutDir(i).name);
        break
    end
end

%% Add Toolbox path to matlab path
thisDir = fileparts(which('KAIST_showTestResults.m'));
piotrToolboxPath = [thisDir, '/../piotr-toolbox-3.40/'];
addpath( genpath( piotrToolboxPath ) );



%% Call function to show results
evalKAIST(inOutDir, dataDir, bbsNmFile, 1)
