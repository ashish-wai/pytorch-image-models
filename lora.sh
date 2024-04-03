model_checkpoint="google/vit-base-patch16-224"
data_dir="/bucket/npss/CottonPestClassification_v3a_npss/"
batch_size=128
learning_rate=2e-5
num_train_epochs=100
fp16=true
output_dir="/bucket/experiments_ashish"

# LoRA parameters
use_lora=true
lora_r=16
lora_alpha=16
lora_dropout=0.1
lora_bias="none"
lora_modules_to_save="classifier"
lora_target_modules="query value"

push_to_hub=true
username="ashishp-wiai"
repository_name="vit-base-patch16-224-finetuned-lora-CottonPestClassification_v3a_npss"

height=224
width=224

experiment="NPSS-Cotton"
run="ViT-LoRA-demo-300epch"


echo "Running python lora.py with all the variables as arguments"
python lora.py \
    --model_checkpoint $model_checkpoint \
    --data_dir $data_dir --batch_size $batch_size \
    --learning_rate $learning_rate --num_train_epochs $num_train_epochs \
    --fp16 $fp16 --use_lora $use_lora --lora_r $lora_r \
    --lora_alpha $lora_alpha --lora_dropout $lora_dropout \
    --lora_bias $lora_bias --lora_modules_to_save $lora_modules_to_save \
    --lora_target_modules $lora_target_modules \
    --push_to_hub $push_to_hub --repository_name $repository_name \
    --height $height --width $width --username $username --output_dir $output_dir \
    --experiment $experiment --run $run

# print command too 
echo "python lora.py \
    --model_checkpoint $model_checkpoint \
    --data_dir $data_dir --batch_size $batch_size \
    --learning_rate $learning_rate --num_train_epochs $num_train_epochs \
    --fp16 $fp16 --use_lora $use_lora --lora_r $lora_r \
    --lora_alpha $lora_alpha --lora_dropout $lora_dropout \
    --lora_bias $lora_bias --lora_modules_to_save $lora_modules_to_save \
    --lora_target_modules $lora_target_modules \
    --push_to_hub $push_to_hub --repository_name $repository_name \
    --height $height --width $width --username $username --output_dir $output_dir \
    --experiment $experiment --run $run"