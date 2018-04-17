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
  "addpath(genpath('~/code/temporalDetection_-EIGEN-/piotr-toolbox-3.40/')); savepath;"
  to add the toolbox to the MATLAB search path.
