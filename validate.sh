#!/bin/bash

# Define variables with desired values (modify as needed)
data_dir="/bucket/npss/CottonPestClassification_v3a_npss/"
split="holdout"
checkpoint="output/train/vit_base_patch16_224.orig_in21k-timm-050524-OS/model_best.pth.tar"
results_file="results/vit_base_patch16_224.orig_in21k-timm-050524-OS_holdout_summary.csv"
model_name="vit_base_patch16_224.orig_in21k"
img_size=224
num_classes=3
mean=(0.485 0.456 0.406)  # List for mean values
std=(0.229 0.224 0.225)   # List for std values

# Construct the command with variables
command="python validate.py"
command+=" --data-dir $data_dir"
command+=" --split $split"
command+=" --checkpoint $checkpoint"
command+=" --results-file $results_file"
command+=" --model $model_name"
command+=" --img-size $img_size"
command+=" --num-classes $num_classes"

# Add mean and standard deviation arguments (assuming separate arguments)
command+=" --mean ${mean[@]}"  # Expand the mean list
command+=" --std ${std[@]}"   # Expand the std list
command+=" --no-prefetcher" # Add flag to disable prefetcher

# Print the constructed command for verification (optional)
echo "Running command:"
echo "$command"

# Run the command
eval "$command"
