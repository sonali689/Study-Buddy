import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
import gc
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(ROOT)
from utils.config_loader import ConfigLoader

class QLoRATrainer:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.fine_tuning_config = config.config['fine_tuning']
        self.model = None
        self.tokenizer = None
        self.lora_config = None
        
    def setup_quantization(self):
        """Setup 4-bit quantization for maximum memory efficiency"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,  # Double quantization for even more memory savings
            bnb_4bit_quant_type="nf4",       # Normalized float 4
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_storage_dtype=torch.uint8,
        )
        return bnb_config

    def clear_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def load_model_and_tokenizer(self):
        """Load base model and tokenizer with 4-bit quantization"""
        model_name = self.fine_tuning_config['base_model']
        
        print(f"🔧 Loading model with 4-bit quantization: {model_name}")
        
        # Clear memory first
        self.clear_memory()
        
        # Setup quantization
        bnb_config = self.setup_quantization()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with 4-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print(" Model loaded with 4-bit quantization")
        
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
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, self.lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        print(" LoRA configuration applied")

    def load_and_tokenize_dataset(self):
        """Load and tokenize dataset with memory efficiency"""
        dataset_path = self.fine_tuning_config['dataset_path']
        
        print(f" Loading dataset from {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            raw_data = json.load(f)
        
        # Format examples with chat template
        formatted_texts = []
        for i, example in enumerate(raw_data):
            try:
                messages = [
                    {
                        "role": "system", 
                        "content": "You are an expert at converting text solutions to properly formatted LaTeX. Format all math with $...$ and $$...$$, use appropriate environments, and ensure compilable code."
                    },
                    {
                        "role": "user", 
                        "content": f"Convert this solution to LaTeX:\n\n{example['text_solution']}"
                    },
                    {
                        "role": "assistant", 
                        "content": example['latex_solution']
                    }
                ]
                
                formatted_text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                formatted_texts.append(formatted_text)
                
            except Exception as e:
                print(f" Skipping example {i} due to error: {e}")
                continue
        
        print(f" Formatted {len(formatted_texts)} examples")
        
        # Create dataset in chunks to avoid memory issues
        def chunk_list(lst, chunk_size):
            for i in range(0, len(lst), chunk_size):
                yield lst[i:i + chunk_size]
        
        # Tokenize in chunks
        all_input_ids = []
        all_attention_mask = []
        
        chunk_size = 100  # Process 100 examples at a time
        for chunk in chunk_list(formatted_texts, chunk_size):
            tokenized = self.tokenizer(
                chunk,
                truncation=True,
                padding=False,
                max_length=self.fine_tuning_config['training']['max_seq_length'],
                return_tensors=None,
            )
            
            all_input_ids.extend(tokenized['input_ids'])
            all_attention_mask.extend(tokenized.get('attention_mask', [[1]*len(ids) for ids in tokenized['input_ids']]))
            
            # Clear memory after each chunk
            self.clear_memory()
        
        # Create dataset dictionary
        dataset_dict = {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        
        print(f" Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")
        
        return train_dataset, eval_dataset

    def setup_training_arguments(self):
        """Setup training arguments optimized for QLoRA"""
        training_params = self.fine_tuning_config['training']
        
        # More conservative settings for QLoRA
        training_args = TrainingArguments(
            output_dir=training_params['output_dir'],
            num_train_epochs=training_params['num_train_epochs'],
            per_device_train_batch_size=1,  # Keep at 1 for QLoRA
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,  # Increase gradient accumulation
            gradient_checkpointing=True,    # Enable gradient checkpointing
            learning_rate=float(training_params['learning_rate']),
            warmup_steps=int(training_params['warmup_steps']),
            logging_steps=int(training_params['logging_steps']),
            save_steps=int(training_params['save_steps']),
            eval_steps=int(training_params['eval_steps']),
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard to save memory
            fp16=True,       # Use mixed precision
            optim="paged_adamw_8bit",  # Use paged optimizer
            lr_scheduler_type="cosine",
            max_grad_norm=0.3,
            dataloader_pin_memory=False,  # Reduce memory usage
            remove_unused_columns=False,
        )
        
        return training_args

    def train(self):
        """Run QLoRA fine-tuning with memory optimization"""
        try:
            # Step 1: Load model with quantization
            self.load_model_and_tokenizer()
            
            # Step 2: Load and tokenize dataset
            train_dataset, eval_dataset = self.load_and_tokenize_dataset()
            
            # Step 3: Setup training arguments
            training_args = self.setup_training_arguments()
            
            # Step 4: Setup data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8,  # Optimize for tensor cores
            )
            
            # Step 5: Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            
            print(" Starting QLoRA fine-tuning...")
            print(" Training with 4-bit quantization and gradient checkpointing")
            
            # Step 6: Start training
            trainer.train()
            
            # Step 7: Save model
            final_output_dir = os.path.join(training_args.output_dir, "final")
            trainer.save_model(final_output_dir)
            self.tokenizer.save_pretrained(final_output_dir)
            
            # Save LoRA config separately
            self.lora_config.save_pretrained(final_output_dir)
            
            print(f" Training complete! Model saved to {final_output_dir}")
            
            return final_output_dir
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(" CUDA Out of Memory! Try these solutions:")
                print("1. Reduce batch size or sequence length")
                print("2. Increase gradient_accumulation_steps")
                print("3. Use a smaller model")
                print("4. Add more GPU memory")
            raise e
        finally:
            # Always clear memory
            self.clear_memory()

def main():
    config = ConfigLoader("config/fine_tuning_config.yaml")
    trainer = QLoRATrainer(config)
    
    try:
        model_path = trainer.train()
        print(f" QLoRA fine-tuning completed successfully!")
        print(f" Model saved at: {model_path}")
        
        # Print memory stats
        if torch.cuda.is_available():
            print(f" GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f" GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
    except Exception as e:
        print(f" QLoRA fine-tuning failed: {e}")
        raise

if __name__ == "__main__":
    main()
