% This script can be used to display and analyze the detection results.
% 
% IMPORTANT: KAIST_test.py has to be executed first, in order to generate
%            all necessary files for this script!


%% Set up necessary path variables
%##########################################################################
% Change "atWORK" to switch between HOME and WORK directories (False: HOME - True: WORK)
atWork = true;

iterToTest     = '40000';  % Specify which solver iteration Nr. to test
thrShow        = 45;  % Set confidence threshold for showing detections

subsetName     = 'train-all-T';
job_name       = 'refinedet_50home_320x320';  % DEFAULT: 'refinedet_vgg16_320x320'
experimentName = 'singleScale';  % Name of evalOutput subfolder (experiment name)

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
evalKAIST(inOutDir, dataDir, bbsNmFile, 1, 'thrShow',thrShow)
