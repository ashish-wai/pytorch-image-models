
import transformers
import accelerate
import peft
import datetime
import warnings
warnings.filterwarnings("ignore")
project_name = "NPSS_Cotton"
import os
os.environ["WANDB_PROJECT"] = project_name
print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")

model_checkpoint = "google/vit-base-patch16-224"

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

ImageNet_mean = [0.5, 0.5, 0.5] 
ImageNet_std = [0.5, 0.5, 0.5] 
normalize = Normalize(mean=ImageNet_mean, std=ImageNet_std)
train_transforms = Compose(
    [
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(224),
        CenterCrop(224),
        ToTensor(),
        normalize,
    ]
)


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


from loaders.ImageData import ImageDataset
data_dir = "/bucket/npss/CottonPestClassification_v3a_os_lora/"

train_loader = ImageDataset(data_dir, split='train', transform=train_transforms)
val_loader = ImageDataset(data_dir, split='val', transform=val_transforms)

label2id = train_loader.labels2id
id2label = train_loader.id2label


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)


from transformers import TrainingArguments, Trainer


model_name = model_checkpoint.split("/")[-1]
batch_size = 128

args = TrainingArguments(
    f"output/{model_name.split('/')[-1]}-finetune-os100_final",
    remove_unused_columns=False,
    learning_rate=5e-3,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    num_train_epochs=50,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
    label_names=["labels"],
    dataloader_num_workers=4,
    dataloader_prefetch_factor=1,
    save_total_limit=100,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    run_name=f"{model_name}-finetune-os100_final",
    save_safetensors=False,
)


import numpy as np
import evaluate

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["image"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


trainer = Trainer(
    model,
    args,
    train_dataset=train_loader,
    eval_dataset=val_loader,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
train_results = trainer.train()