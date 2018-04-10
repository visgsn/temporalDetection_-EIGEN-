#!/usr/bin/env bash

# You can modify the parameters in create_data.sh if needed.
# It will create lmdb files for trainval and test with encoded original image:
#   - $HOME/data/KAIST/<trainsetName>/ImageSets/lmdb/KAIST_trainval_lmdb
#   - $HOME/data/KAIST/<trainsetName>/ImageSets/lmdb/KAIST_test_lmdb
# and make soft links at examples/VOC0712/

##### CONFIGURATION ############################################################
redo=1
dataset_name="KAIST"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0
extra_cmd="--encode-type=png --encoded"

### *** HOME ***
RefineDet_ROOT="$HOME/code/caffe/RefineDet"
tmpDetRepo_ROOT="$HOME/code/temporalDetection_(EIGEN)"
data_root_dir="$HOME/data/KAIST"
### *** WORK ***
#RefineDet_ROOT="$HOME/code/caffe/RefineDet"
#tmpDetRepo_ROOT="$HOME/code/temporalDetection_(EIGEN)"
#data_root_dir="/net4/merkur/storage/deeplearning/users/gueste/data/KAIST/"     # CHECK this!!!

trainsetName="train-all-T"
mapfile="$tmpDetRepo_ROOT/KAIST_preparation/labelmap_KAIST.prototxt"
################################################################################


cd $RefineDet_ROOT

if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in trainval test
do                                                                                                                                                                                                               #--root_dir    #--list_file                                            #--out_dir                                                               #--example_dir
  python $RefineDet_ROOT/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $data_root_dir/$trainsetName/ImageSets/Main/$subset.txt $data_root_dir/$trainsetName/ImageSets/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name/$trainsetName
done
