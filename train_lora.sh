#!/bin/bash

# Define variables with desired values (modify as needed)
dataset_path="/bucket/npss/CottonPestClassification_v3a_npss/"
train_split="train"
img_size=224
val_split="val"
model_name="google/vit-base-patch16-224-in21k"
use_pretrained=true  # set to true or false
output_path="/bucket/experiments_ashish"
name="NPSS_Cotton"
experiment_name="ViT_LoRA_Trials"
use_wandb=true      # set to true or false
device="cuda"  # can be "cpu" or "cuda" (if GPU available) default is "cuda"
data_path="/bucket/npss/CottonPestClassification_v3a_npss/"
batch_size=128
epochs=300  
mean=(0.485 0.456 0.406) # ImageNet mean
std=(0.229 0.224 0.225) # ImageNet std
num_classes=3
mapping="Cotton_ClassMap.txt"
# validation_batch_size=None
## LORA Params
lora=true
lora_r=16
lora_alpha=16
lora_dropout=0.1
lora_bias="none"
lora_modules_to_save="classifier"
target_modules="query value"

# Construct the command with variables
command="python train_lora.py"
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

# LORA Params
if [ $lora == true ]; then
  command+=" --lora"
fi
command+=" --lora-r $lora_r"
command+=" --lora-alpha $lora_alpha"
command+=" --lora-dropout $lora_dropout"
command+=" --lora-bias $lora_bias"
command+=" --lora-modules-to-save $lora_modules_to_save"
command+=" --lora-target-modules $target_modules"

# Print the constructed command for verification (optional)
echo "Running command: $(date)"
echo "$command"
echo "===================="

# Run the command
eval "$command"
