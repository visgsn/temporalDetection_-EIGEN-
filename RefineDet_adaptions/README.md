To enable own layers in RefineDet structure, it's necessary to replace certain
(original) files in the Refinedet-Code with symlinks to adapted files in the
"temporalDetection_-EIGEN-" folder.
These files contain basically the same code but are extended to fulfill
all necessary conditions for any temporal extension provided in this repo.



To establish symlinks to the original files, do the following:

1.	Backup the original file e.g. with:
   	mv -nv <sourceFileName> <sourceFileName_ORIGINAL>

2.	Create symlinks to replace the original files with our new (adapted) ones:
	ln -sirv <newFile> <originalFileParentDirectory>
	
	Example:
	ln -sirv model_libs.py /home/gueste/code/caffe/RefineDet/python/caffe/



FILES TO REPLACE (with links):

- model_libs.py --> /home/gueste/code/caffe/RefineDet/python/caffe/
