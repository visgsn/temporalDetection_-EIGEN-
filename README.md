# temporalDetection_-EIGEN-
Code from Master's thesis in 2018.

In order to get the Code Working, install prerequisits and follow the instructions listed below:
- Install RefineDet detector according to the given installation instructions (link below).
  https://github.com/sfzhang15/RefineDet

- Install MATLAB

- The recommended path for this repository is "~/code/temporalDetection_-EIGEN-"

- Copy the lib folder from "<RefineDet_root>/test/" into "temporalDetection_-EIGEN-/evaluationScripts/DEMO/".
  (please don't forget to adapt all paths according to the instructions, mentioned in the "Evaluation" chapter of the
  RefineDet README.md!)

- Adapt all paths in the scripts you want to execute, too. Especially the paths for datasets to use or your
  desired output folders.

- In order to use the MATLAB evaluation scripts with Piotrs toolbox (already included in this repo!)
  sinply open MATLAB and execute:
  1. "addpath(genpath('~/code/temporalDetection_-EIGEN-/piotr-toolbox-3.40/')); savepath;"
  OR
  2. "addpath(genpath('~/code/temporalDetection_-EIGEN-/piotr-toolbox-3.40/')); savepath '~/code/temporalDetection_-EIGEN-/piotr-toolbox-3.40/pathdef.m';"
  if matlab throws a warning while saving the new MATLAB path in the default directory.
  (In case of the second option (2.), you have to start MATLAB from the piotr-toolbox folder every time in order to use the new file correctly.)
  This adds the toolbox to the MATLAB search path.
  
  Aditionally install the "MATLAB Engine", since it's needed by the evaluation scripts. Just follow the instructions on the link below:
  https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

- Replace original RefineDet files with adapted ones from ".../temporalDetection_-EIGEN-/RefineDet_adaptions/":
  Detailed instructions for each file can be found there, too (see README.md file).
