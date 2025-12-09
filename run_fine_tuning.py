"""
Main script to run the fine-tuning pipeline
"""

import os
import argparse
from fine_tuning.train_lora import main as train_lora

def main():
    parser = argparse.ArgumentParser(description="Run LaTeX Formatter Fine-tuning")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON file")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2", help="Base model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    
    args = parser.parse_args()
    
    print(" Starting Fine-tuning Pipeline")
    print("=" * 50)
    
    # Step 1: Prepare dataset
    print("\n1. Preparing dataset...")
    # This would prepare and format the dataset
    
    # Step 2: Run fine-tuning
    print("\n2. Starting LoRA fine-tuning...")
    try:
        train_lora()
        print(" Fine-tuning completed successfully!")
    except Exception as e:
        print(f" Fine-tuning failed: {e}")
        return
    
    print("\n Fine-tuning pipeline complete!")
    print(" Fine-tuned model saved in: ./models/formatter_fine_tuned/")

if __name__ == "__main__":
    main()
