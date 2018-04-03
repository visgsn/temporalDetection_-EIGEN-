function convertVbb2Txt( fileNameWithPath )
% This function converts the fName.vbb annotation file into .txt format.
% It extracts every frame into one .txt file.

%Generate path for output folder
[PATHSTR,NAME,~] = fileparts(fileNameWithPath);
tarDir = join([PATHSTR, '/', NAME]);

% Load annotation from disk
A = vbb('vbbLoad', fileNameWithPath)
% Export single frame annotations to tarDir/*.txt
vbb( 'vbbToFiles', A, tarDir)

end