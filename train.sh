#!/bin/bash

# Define variables with desired values (modify as needed)
dataset_path="/bucket/npss/CottonPestClassification_v3a_os/"
train_split="train"
img_size=224
val_split="val"
model_name="vit_base_patch16_224"
pretrained=false # set to true or false
pretrained_path=""
use_pretrained=true  # set to true or false
# output_path="/bucket/weights"
name="NPSS_Cotton"
experiment_name="ViT_Trials_OS"
use_wandb=true      # set to true or false
device="cuda"  # can be "cpu" or "cuda" (if GPU available) default is "cuda"
data_path="/bucket/npss/CottonPestClassification_v3a_os/"
batch_size=128
epochs=300  
mean=(0.485 0.456 0.406) # ImageNet mean
std=(0.229 0.224 0.225) # ImageNet std
num_classes=3
mapping="Cotton_ClassMap.txt"
# validation_batch_size=None

# Construct the command with variables
command="python train.py"
command+=" --train-split $train_split"
command+=" --val-split $val_split"
command+=" --model $model_name"
if [ $use_pretrained == true ]; then
  command+=" --pretrained"
fi
# command+=" --output $output_path"
command+=" --experiment $experiment_name"
if [ $use_wandb == true ]; then
  command+=" --log-wandb"
fi
if [ $pretrained == true ]; then
  command+=" --pretrained "
  command+=" --pretrained-path $pretrained_path"
fi
command+=" --device $device"
command+=" --data-dir $data_path"
command+=" --batch-size $batch_size"
command+=" --epochs $epochs"
command+=" --mean ${mean[@]}"
command+=" --std ${std[@]}"
command+=" --num-classes $num_classes"
command+=" --mapping $mapping"
command+=" --img-size $img_size"
command+=" --name $name"
# Print the constructed command for verification (optional)
echo "Running command:"
echo "$command"

# Run the command
eval "$command"
