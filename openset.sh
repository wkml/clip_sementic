#!/bin/bash
post='new_SD-exp3.1-cat_global_local'
backbone_name='RN101'
dataset='COCO'
train_data_dir='/data/public/coco2014/train2014'
train_list='/data/public/coco2014/annotations/instances_train2014.json'
test_data_dir='/data/public/coco2014/val2014'
test_list='/data/public/coco2014/annotations/instances_val2014.json'
category_file='./data/coco/category_name.json'

num_classes=80
batch_size=256
epochs=20

learning_rate=0.001
momentum=0.9
weight_decay=0.005

#input parameter
crop_size=448
scale_size=512

#number of data loading workers
workers=5

#manual epoch number (useful on restarts)
start_epoch=0
print_freq=50


N_CTX=16
# CTX_INIT=
CLASS_TOKEN_POSITION=end

cuda=0
CUDA_VISIBLE_DEVICES=${cuda} python new_sd.py \
--dataset ${dataset} \
--train_data ${train_data_dir} \
--test_data ${test_data_dir} \
--train_list ${train_list} \
--test_list ${test_list} \
--batch_size ${batch_size} \
--workers ${workers} \
--epochs ${epochs} \
--start_epoch  ${start_epoch} \
--batch_size ${batch_size} \
--learning-rate ${learning_rate} \
--momentum ${momentum} \
--weight_decay ${weight_decay} \
--crop_size ${crop_size} \
--scale_size ${scale_size} \
--print_freq ${print_freq} \
--post ${post} \
--backbone_name ${backbone_name} \
--class_token_position ${CLASS_TOKEN_POSITION} \
--n_ctx ${N_CTX} \
--openset \