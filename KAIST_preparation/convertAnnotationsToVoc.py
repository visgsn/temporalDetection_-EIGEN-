'''
    This script converts all annotations from the KAIST dataset to a compatible format for VOC
    and exports them into the output folder.
'''

from dirRecursive import dirRecursive
from fileParts import fileParts


### *** HOME ***                                                                    # anpassen!
#kaistFolder = '/home/herrma/deepdata/datasets/KAIST/data-kaist/'
#outFolder = '/home/herrma/deepdata/users/herrma/datasets/KAIST/Annotations/'
### *** WORK ***                                                                    # anpassen!
kaistFolder = '/home/herrma/deepdata/datasets/KAIST/data-kaist/'
outFolder = '/home/herrma/deepdata/users/herrma/datasets/KAIST/Annotations/'

excludeLabels = ['people','person?','cyclist']                                      # persons sollen rein?


for fold in ['train-all20','test-all']:
    annoFiles = dirRecursive(kaistFolder + fold + '/annotations', '*.txt')
    imgFiles = dirRecursive(kaistFolder + fold + '/images', 'T_*.png')              # "T_" --> Thermal
    
    for (i,fPath) in enumerate(annoFiles):
        _,imgName,_ = fileParts(imgFiles[i])
        with open(fPath) as f:
            
            with open(outFolder + imgName + '.xml', 'w') as outFile:
                # write header
                outFile.write('<annotation>\n\t<folder>KAIST</folder>\n')
                outFile.write('\t<filename>' + imgName + '.png' + '</filename>\n')
                outFile.write('\t<source>\n\t\t<database>KAIST</database>\n')
                outFile.write('\t\t<annotation>KAIST Annotation</annotation>\n')
                outFile.write('\t\t<image>' + imgName + '</image>\n\t</source>\n')
                outFile.write('\t<size>\n\t\t<width>640</width>\n')
                outFile.write('\t\t<height>512</height>\n')
                outFile.write('\t\t<depth>3</depth>\n\t</size>\n')                 # Abmessungen korrekt? (depth)
                outFile.write('\t<segmented>0</segmented>\n')
                   
                # write bounding boxes                 
                for line in f:
                    if '%' not in line:   
                        parts = line.split(' ')
                        
                        if parts[0] not in excludeLabels:
                            outFile.write('\t<object>\n\t\t')
                            outFile.write('<name>' + parts[0] + '</name>\n\t\t')
                            outFile.write('<bndbox>\n\t\t\t')
                            outFile.write('<xmin>' + parts[1] + '</xmin>\n\t\t\t')
                            outFile.write('<ymin>' + parts[2] + '</ymin>\n\t\t\t')
                            outFile.write('<xmax>{:d}</xmax>\n\t\t\t'.format(int(parts[1]) + int(parts[3]) - 1))
                            outFile.write('<ymax>{:d}</ymax>\n\t\t\t'.format(int(parts[2]) + int(parts[4]) - 1))
                            outFile.write('</bndbox>\n\t\t')
                            outFile.write('<difficult>0</difficult>\n\t')
                            outFile.write('</object>\n')                               
                                
                outFile.write('</annotation>')
