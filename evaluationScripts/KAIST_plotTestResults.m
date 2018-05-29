% This script can be used to plot the detection results of 
% different experiments and iterations together in one
% single plot (log-average miss rate).
% 
% IMPORTANT: KAIST_test.py has to be executed first, in order to generate
%            all necessary files for this script!


%% Set up necessary path variables
%##########################################################################
% Change "atWORK" to switch between HOME and WORK directories (False: HOME - True: WORK)
atWork = false;

subsetName = 'train-all-T';
job_name   = 'Tr11_HOME_i10k_Adam_512x512';  % DEFAULT: 'refinedet_vgg16_320x320'
% Optional: Use the following path directly with 'useDirectJobPath = true'
useDirectJobPath = false;
directJobPath = ...
    '/home/gueste/train_test_data/models/VGGNet/KAIST/train-all-T/Tr10_i40k_Adam_DROP_lr0001STEP_512x512';

experimentNames = ...  % Name of evalOutput subfolder (experiment name)
    ["singleScale", "multiScale"];

% Path prefix for output directory
path_prefix_HOME = sprintf('%s/train_test_data', getenv('HOME'));
path_prefix_WORK = '/net4/merkur/storage/deeplearning/users/gueste/TRAINING_test';
% Result file to read
inputFileName    = 'mAP_mPrec_mRec_laMiss.txt';
%##########################################################################

%% Construct basic paths from config
if useDirectJobPath == false
    if atWork == true
        path_prefix = path_prefix_WORK;
    else
        path_prefix = path_prefix_HOME;
    end

    train_test_outPath = sprintf(...
        '%s/models/VGGNet/KAIST/%s/%s',...
        path_prefix, subsetName, job_name);
else
    train_test_outPath = directJobPath;
end
% Check paths
if ( ~exist(train_test_outPath, 'dir') )
    error('KAIST_showTestResults:DirectoryNotFound', ...
        'Path for train_test_outPath does not exist! --> atWORK = ?!');
end

outDir = sprintf(...
    '%s/evalOutput/', ...
    train_test_outPath);

%% Add Toolbox path to matlab path
thisDir = fileparts(which('KAIST_plotTestResults.m'));
piotrToolboxPath = [thisDir, '/../piotr-toolbox-3.40/'];
addpath( genpath( piotrToolboxPath ) );

%% Prepare figure for result plot
f = figure();
axis([-inf inf 0 100]); grid on;
title("training progress: log-average miss rate");
xlabel("iterations"); ylabel("LAMR [%]");
hold on;
legend on;

%% Iterate through all experiments
for expNameIter = 1:length(experimentNames)
    %% Construct complete input paths
    inDir = sprintf(...
        '%s%s/', ...
        outDir, experimentNames(expNameIter));
    % Check if path exists
    if ( ~exist(inDir, 'dir') )
        fprintf( ...
            ['WARNING: Path for experiment named %s does not exist!', ...
            ' --> Run KAIST_test.py?!\n'], experimentNames(expNameIter));
    end
    %% Search for correct input file with person detections
    filesInOutDir = dir(inDir);  % List all files in inDir
    for i = 1:length(filesInOutDir)
        if contains(filesInOutDir(i).name, inputFileName) && ...
                    filesInOutDir(i).isdir == 0
            resultFile = sprintf(...
                '%s/%s', filesInOutDir(i).folder, filesInOutDir(i).name);
            fprintf('Found File: %s\n', resultFile);
            break
        end
    end
    
    %% Load resultFile data
    % resultValues = [iter, mAP, mPrec, mRec, lamr]
    resultValues = importResultFile(resultFile);
    % Rank according to lamr values
    resultValSorted = sortrows(resultValues, 'lamr');

    %% Plot results
    plot(resultValues.iter, resultValues.lamr * 100, '-+', ...
        'DisplayName', experimentNames(expNameIter), 'LineWidth', 2);
    % Mark best datapoints (minimal LAMR)
    plot(resultValSorted.iter(1), resultValSorted.lamr(1) * 100, 'd', ...
        'DisplayName', ...
        sprintf('--> Top_{LAMR} = %.1f%% at %d', ...
                resultValSorted.lamr(1)*100, ...
                resultValSorted.iter(1)), ...
        'LineWidth', 2, 'MarkerSize', 8);
end

%% Save results as figure and image
outputFileName = [outDir, 'LAMR_results_plotted'];
savefig([outputFileName '.fig']);
frame=getframe(f);
[X,~]=frame2im(frame);
imwrite(X,[outputFileName '.png'], 'png');
% Print info message
fprintf('Saved output files to: %s%s\n', outputFileName, '.<ext>');
close(f);  % Close old figure

