# You can modify the parameters in create_data.sh if needed.
# It will create lmdb files for trainval and test with encoded original image:
#   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb
#   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb
# and make soft links at examples/VOC0712/

cd $RefineDet_ROOT
cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir/../..
cd $root_dir

##### CONFIGURATION ############################################################
redo=1
dataset_name="KAIST"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0
extra_cmd="--encode-type=jpg --encoded"

### *** HOME ***
RefineDet_ROOT  = "$HOME/code/caffe/RefineDet"
data_root_dir   = "$HOME/data/KAIST"
mapfile         = "$root_dir/data/$dataset_name/labelmap_voc.prototxt"          # Has to be adapted!!!
### *** WORK ***
#RefineDet_ROOT  = "$HOME/code/caffe/RefineDet"
#data_root_dir   = "/net4/merkur/storage/deeplearning/users/gueste/data/VOCdevkit/"
#mapfile         = "$root_dir/data/$dataset_name/labelmap_voc.prototxt"          # Has to be adapted!!!
################################################################################

if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in trainval test
do
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/data/$dataset_name/$subset.txt $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name
done
