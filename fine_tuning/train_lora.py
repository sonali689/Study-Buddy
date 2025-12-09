import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
from utils.config_loader import ConfigLoader

class LoRATrainer:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.fine_tuning_config = config.config['fine_tuning']
        self.model = None
        self.tokenizer = None
        self.lora_config = None
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer"""
        model_name = self.fine_tuning_config['base_model']
        
        print(f" Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA configuration
        lora_params = self.fine_tuning_config['lora']
        self.lora_config = LoraConfig(
            r=lora_params['r'],
            lora_alpha=lora_params['lora_alpha'],
            lora_dropout=lora_params['lora_dropout'],
            target_modules=lora_params['target_modules'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
    
    def load_dataset(self):
        """Load and tokenize dataset"""
        dataset_path = self.fine_tuning_config['dataset_path']
        
        print(f" Loading dataset from {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            raw_data = json.load(f)
        
        # Format examples
        formatted_texts = []
        for example in raw_data:
            messages = [
                {"role": "system", "content": "Convert text solutions to LaTeX format."},
                {"role": "user", "content": f"Convert to LaTeX: {example['input']}"},
                {"role": "assistant", "content": example['output']}
            ]
            formatted_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            formatted_texts.append(formatted_text)
        
        # Create dataset
        dataset = Dataset.from_dict({"text": formatted_texts})
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.fine_tuning_config['training']['max_seq_length'],
                return_tensors=None,
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        return tokenized_dataset
    
    def train(self):
        """Run LoRA fine-tuning"""
        # Load model and dataset
        self.load_model_and_tokenizer()
        tokenized_dataset = self.load_dataset()
        
        # Split dataset
        train_size = int(0.9 * len(tokenized_dataset))
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
        
        print(f"📊 Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")
        
        # Training arguments
        training_params = self.fine_tuning_config['training']
        training_args = TrainingArguments(
            output_dir=training_params['output_dir'],
            num_train_epochs=training_params['num_train_epochs'],
            per_device_train_batch_size=training_params['per_device_train_batch_size'],
            per_device_eval_batch_size=training_params['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_params['gradient_accumulation_steps'],
            learning_rate=training_params['learning_rate'],
            warmup_steps=training_params['warmup_steps'],
            logging_steps=training_params['logging_steps'],
            save_steps=training_params['save_steps'],
            eval_steps=training_params['eval_steps'],
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard if not needed
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        print(" Starting LoRA fine-tuning...")
        
        # Start training
        trainer.train()
        
        # Save final model
        final_output_dir = os.path.join(training_params['output_dir'], "final")
        trainer.save_model(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)
        
        print(f" Training complete! Model saved to {final_output_dir}")
        
        return final_output_dir

def main():
    config = ConfigLoader("config/fine_tuning_config.yaml")
    trainer = LoRATrainer(config)
    
    try:
        model_path = trainer.train()
        print(f" Fine-tuning completed successfully!")
        print(f" Model saved at: {model_path}")
    except Exception as e:
        print(f" Fine-tuning failed: {e}")
        raise

if __name__ == "__main__":
    main()
