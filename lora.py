import argparse
import configparser
import transformers
import accelerate
import peft
from loaders.ImageData import ImageDataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from evaluate import load
from peft import LoraConfig, get_peft_model
import json
import numpy as np
import yaml
import torch
import evaluate


metric = evaluate.load("accuracy")

def get_config(config_path=None):
    if config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    parser = argparse.ArgumentParser(description="Fine-tune ViT model with LoRA")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path or name of the pre-trained model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="Learning rate for the optimizer")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of epochs to train the model")
    parser.add_argument("--fp16", type=bool, default=False, help="Whether to use mixed precision training")
    parser.add_argument("--push_to_hub", type=bool, default=False, help="Whether to push the model to the Hugging Face Hub")
    parser.add_argument("--repository_name", type=str, help="Name of the repository to push the model to")
    parser.add_argument("--output_dir", type=str, help="Path to save the fine-tuned model")
    parser.add_argument("--use_lora", type=bool, default=False, help="Whether to use LoRA")
    parser.add_argument("--lora_r", type=int, default=1, help="Number of attention heads to keep active")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="Alpha value for LoRA")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", help="List of target modules for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate for LoRA")
    parser.add_argument("--lora_bias", type=str, help="Whether to use bias in LoRA")
    parser.add_argument("--lora_modules_to_save", type=str, nargs="+", help="List of modules to save for LoRA")
    parser.add_argument("--height", type=int, default=224, help="Height of the input image")
    parser.add_argument("--width", type=int, default=224, help="Width of the input image")
    parser.add_argument("--username", type=str, help="Username for the Hugging Face Hub")
    parser.add_argument("--experiment", type=str, help="Name of the experiment")
    parser.add_argument("--run", type=str, help="Name of the run")
    args = parser.parse_args()
    return vars(args)


def load_data(data_dir, split, transform):
    return ImageDataset(data_dir, split=split, transform=transform)


def create_transforms(height, width):
    ImageNet_mean = [0.485, 0.456, 0.406]
    ImageNet_std = [0.229, 0.224, 0.225]
    normalize = Normalize(mean=ImageNet_mean, std=ImageNet_std)

    train_transforms = Compose(
        [
            RandomResizedCrop(height),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    val_transforms = Compose(
        [
            Resize(height),
            CenterCrop(height),
            ToTensor(),
            normalize,
        ]
    )

    return train_transforms, val_transforms


def create_model(config, label2id, id2label):

    model = AutoModelForImageClassification.from_pretrained(
                    config["model_checkpoint"],
                    label2id=label2id,
                    id2label=id2label,
                    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
                )
    # model.classifier.out_proj = torch.nn.Linear(model.classifier.in_features, len(label2id))  # Adjust output layer

    if config.get("use_lora", False):  # Check if LoRA is enabled in config
        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            target_modules=config["lora_target_modules"],
            lora_dropout=config["lora_dropout"],
            bias=config["lora_bias"],
            modules_to_save=config["lora_modules_to_save"],
        )
        model = get_peft_model(model, lora_config)

    return model

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    pixel_values = torch.stack([example["image"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def main(config_path=None):
    config = get_config(config_path)

    # Load data
    train_transforms, val_transforms = create_transforms(config["height"], config["width"])
    train_dataset = load_data(config["data_dir"], "train")
    val_dataset = load_data(config["data_dir"], "val")

    # Create model
    model = create_model(config, train_dataset.labels2id, train_dataset.id2label)

    # Define training arguments
    training_args = TrainingArguments(
        f"{config['output_dir']}/{config['experiment']}/{config['run']}",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=config["batch_size"],
        fp16=config["fp16"],
        num_train_epochs=config["num_train_epochs"],
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=config["push_to_hub"],
        label_names=list(train_dataset.id2label.values()),
        
    )

    # Define trainer instance
    print("Training model...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=AutoImageProcessor.from_pretrained(config["model_checkpoint"]),
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    # save config and model
    # with open(f"{training_args.output_dir}/config.json", "w") as f:
    #     json.dump(config, f, indent=4)
    

    train_result = trainer.train()

    print("Train result:", train_result)

    if training_args.push_to_hub:
        repository_name = config["repository_name"] if config["repository_name"] \
            else f"{config['model_checkpoint'].split('/')[-1]}-finetuned-lora-{config['data_dir'].split('/')[-1]}"
        username = config["username"]
        print(f"Pushing model to the Hugging Face Hub repository: {username}/{repository_name}")
        trainer.push_to_hub(repository_name=username+"/"+repository_name)


        
if __name__ == "__main__":
    main()