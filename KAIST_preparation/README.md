This Readme describes how to use the "KAIST_preparation" scripts for preparation
of the KAIST dataset.
It should then be ready to use with the RefineDet detector.
(Requirements: Matlab has to be installed, together with the "Pjotr"- and
               "Seq_2_Avi"-Toolbox)
IMPORTANT: Check all paths inside the scripts before executing any of it!


Please execute the following steps in the given order:

1.  convertVbb2Txt.m:
    Use this script to convert the .vbb annotation files in
    '${KAIST_root}/data-kaist/annotations' to single frame annotations.
    (Current status is to convert the desired .vbb files manually, file by file)

2.  copyAndRename.py:
    This script is used to copy and rename the images and annotations from the KAIST dataset, which have previously
    been extracted with Pjotr's Matlab Toolbox. (By default only training data from set00 to set05)
    Names are chosen according to the scheme of "train_all20" with one exception - instead of the unique ID at the
    end of the generated file names, the last element of 'outputDir' is used as postfix (e.g. _train-all).
    Example for images:
        T_tmp_set00_V000_I00019_train-all.png         (T_ stands for "thermal", RGB_ for color images)
    Example for annotations:
        set00_V000_I00019_train-all.txt

3.  convertAnnotationsToVoc.py:
    This script converts all annotations from the KAIST dataset to a compatible format for VOC
    and exports them as .xml files into the output folder (renamed according to image name).
    --> Check paths before execution!

4.  generateVocLists.py:
    This script creates trainval.txt, test.txt and test_name_size.txt, which contain the links between images and their
    corresponding annotations, as well as image size information.
    To use this script, first run "convertAnnotations.py" to convert the necessary Annotations
    from the kaist-data sub-folder into the KAIST folder (target format: .xml).
    IMPORTANT: Check paths below before execution!
               "kaistDataFolder" has to point to the location of the "data-kaist" folder inside KAIST!!!

5.  createData.sh:
    You can modify the parameters in create_data.sh if needed.
    It will create lmdb files for trainval and test with encoded original image:
     - $HOME/data/KAIST/<trainsetName>/ImageSets/lmdb/KAIST_trainval_lmdb
     - $HOME/data/KAIST/<trainsetName>/ImageSets/lmdb/KAIST_test_lmdb
    and make soft links at examples/VOC0712/
