function dbExtract2( pth, type, flatten, skip )
% Extract database to directory of images and ground truth text files.
%
% Call 'dbInfo(name)' first to specify the dataset. The format of the
% ground truth text files is the format defined and used in bbGt.m.
%
% USAGE
%  dbExtract( tDir, flatten )
%
% INPUTS
%  tDir     - [] target dir for image data (defaults to dataset dir)
%  flatten  - [0] if true output all images to single directory
%  skip     - [] specify frames to extract (defaults to skip in dbInfo)
%
% OUTPUTS
%
% EXAMPLE
%  dbInfo('InriaTest'); dbExtract;
%
% See also dbInfo, bbGt, vbb
%
% Caltech Pedestrian Dataset     Version 3.2.1
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

tDir = [ pth type '/' ];
[pth,setIds,vidIds,dbName] = dbInfo2( [], pth );
if(nargin<2 || isempty(tDir)), tDir=pth; end
if(nargin<3 || isempty(flatten)), flatten=0; end
if(nargin<4 || isempty(skip)), [~,~,~,~,skip]=dbInfo2; end

if strcmp( dbName, 'kaist' )
    % create local copy of fName which is in a imagesci/private
    sName = [fileparts(which('imread.m')) filesep 'private' filesep 'pngreadc.mexw64'];
    tName = ['.' filesep 'pngreadc.mexw64'];
    if(~exist(tName,'file')), copyfile(sName,tName); end
end

for s=1:length(setIds)
  for v=1:length(vidIds{s})
    
    % load ground truth
    name=sprintf('set%02d/V%03d',setIds(s),vidIds{s}(v));      
    
    if strcmp( dbName, 'kaist-place' )
	  vSr = seqIo(sprintf('%svideos/set%02d/Visible_V%03d.seq',pth,setIds(s),vidIds{s}(v)), 'reader');
      info= vSr.getinfo();      n = info.numFrames;
    else
      A=vbb('vbbLoad',[pth 'annotations/' name]); n=A.nFrame;
    end
      
    if(flatten), post=''; else post=[name '/']; end
    if(flatten), f=[name '_']; f(6)='_'; else f=''; end
    fs=cell(1,n); for i=1:n, fs{i}=[f 'I' int2str2(i-1,5)]; end
    
    % extract images
    td=[tDir 'images/' post]; if(~exist(td,'dir')), mkdir(td); end
    if strcmp( dbName, 'kaist' ) || strcmp( dbName, 'kaist-place' )
        vSr = seqIo(sprintf('%svideos/set%02d/Visible_V%03d.seq',pth,setIds(s),vidIds{s}(v)), 'reader');
        iSr = seqIo(sprintf('%svideos/set%02d/LWIR_V%03d.seq',pth,setIds(s),vidIds{s}(v)), 'reader');
        info= vSr.getinfo();
        for i=skip-1:skip:n-1
            t=clock; t=mod(t(end),1);
            tNm=sprintf('tmp_%s_%15i.%s',fs{i+1},round((t+rand)/2*1e15),'png');
            fv=[td 'RGB_' tNm];
            fid = fopen(fv,'w'); vSr.seek(i); I = vSr.getframeb(); fwrite(fid,I); fclose(fid);                        
            fi=[td 'T_' tNm];
            fid = fopen(fi,'w'); iSr.seek(i); I = iSr.getframeb(); fwrite(fid,I); fclose(fid);
            
            vI = imread( fv );                iI = imread( fi ); 
            delete( fv );   delete( fi );
            
            f=[td fs{i+1} '.' info.ext];
            if ndims(iI) == 3
                imwrite( cat(3, vI, rgb2gray(iI)), f, 'tiff', 'Compression', 'deflate' );
            else
                imwrite( cat(3, vI, iI), f, 'tiff', 'Compression', 'deflate' );
            end
        end; vSr.close();   iSr.close();            
    else
        sr=seqIo([pth 'videos/' name '.seq'],'reader'); info=sr.getinfo();
        for i=skip-1:skip:n-1
            f=[td fs{i+1} '.' info.ext]; if(exist(f,'file')), continue; end
            sr.seek(i); I=sr.getframeb(); f=fopen(f,'w'); fwrite(f,I); fclose(f);
        end; sr.close();
        
    end
    if ~strcmp( dbName, 'kaist-place' )
        % extract ground truth
        td=[tDir 'annotations/' post];
        for i=1:n, fs{i}=[fs{i} '.txt']; end
        vbb('vbbToFiles',A,td,fs,skip,skip);
    end
  end
end

end
