import torch
import os
import numpy as np
from typing import List, Dict

# Fix imports - use proper Hugging Face imports
from transformers import (
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)

from model import DNATokenizer, DNADataset, MegaDNAConfig, MegaDNACausalLM
from utils import split_dataset, load_dna_sequences


def load_presplit_data(data_dir: str) -> Dict[str, List[str]]:
    """
    Load pre-split train/validation/test data from npy files.
    
    Args:
        data_dir: Directory containing train_sequences.npy, 
                  validation_sequences.npy, test_sequences.npy
    
    Returns:
        Dictionary with 'train', 'validation', 'test' keys containing lists of DNA sequences
    """
    splits = {}
    
    for split_name in ['train', 'validation', 'test']:
        npy_path = os.path.join(data_dir, f'{split_name}_sequences.npy')
        if os.path.exists(npy_path):
            sequences = np.load(npy_path, allow_pickle=True)
            splits[split_name] = sequences.tolist()
            print(f"Loaded {len(splits[split_name])} {split_name} sequences from {npy_path}")
        else:
            splits[split_name] = []
            print(f"Warning: {npy_path} not found, using empty {split_name} set")
    
    return splits



# Enhanced incremental pretraining setup with pre-split datasets
def setup_incremental_pretraining(
    model_path: str,
    tokenizer_path: str,
    dataset_splits: Dict[str, List[str]],  # Changed from training_data to pre-split data
    output_dir: str,
    learning_rate: float = 5e-5,
    train_batch_size: int = 4,
    eval_batch_size: int = 8,
    num_epochs: int = 3,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 8192,
    device: str = 'cuda',
    logging_steps: int = 100,
    save_steps: int = 500,
    eval_steps: int = 500,
    resume_from_checkpoint: str = None,
    **kwargs
):
    """
    Set up and run incremental pretraining for MegaDNA model using pre-split datasets.
    
    Args:
        model_path: Path to pretrained model
        tokenizer_path: Path to tokenizer
        dataset_splits: Dictionary with 'train', 'validation', 'test' keys containing lists of DNA sequences
        output_dir: Output directory for trained model
        ... other training parameters
    """
    
    # Load tokenizer
    if os.path.exists(tokenizer_path):
        tokenizer = DNATokenizer.from_pretrained(tokenizer_path)
    else:
        # Create default tokenizer if path doesn't exist
        tokenizer = DNATokenizer()
        print(f"Tokenizer path {tokenizer_path} not found. Using default tokenizer.")
    
    # Create datasets
    train_dataset = DNADataset(
        dataset_splits['train'], 
        tokenizer, 
        max_length=max_seq_length,
        is_training=True
    )
    
    val_dataset = DNADataset(
        dataset_splits['validation'], 
        tokenizer, 
        max_length=max_seq_length,
        is_training=False
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Load model configuration and model
    # Create default config if loading fails
    config = MegaDNAConfig(
                vocab_size=tokenizer.vocab_size,
                dim=(768, 512, 256),
                depth=(6, 4, 2),
                max_seq_len=(128, 32, 16)
            )
    print(f"Using default configuration.")
    
    # Initialize model
    model = MegaDNACausalLM(config)
    
    # Load pretrained weights if model_path exists
    if os.path.exists(model_path):
        # try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        state_dict = checkpoint.state_dict()
        model.load_state_dict(state_dict, strict=False)
    
    # Move model to device
    model = model.to(device)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        bf16=True,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        **kwargs
    )
    
    # Define compute metrics function
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        
        # Handle cases where logits might be CausalLMOutput or tuple
        if hasattr(logits, 'logits'):
            logits = logits.logits
        elif isinstance(logits, (list, tuple)) and len(logits) > 0:
            logits = logits[0]
            
        # Calculate perplexity as metric
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100) #tokenizer.pad_token_id
        logits_tensor = torch.tensor(logits) if not isinstance(logits, torch.Tensor) else logits
        labels_tensor = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
        
        # Shift logits and labels for causal LM
        shift_logits = logits_tensor[..., :-1, :].contiguous()
        # shift_labels = labels_tensor[..., 1:].contiguous()
        shift_labels = labels_tensor.contiguous()

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = torch.exp(loss)
        
        return {
            'perplexity': perplexity.item(),
            'loss': loss.item()
        }
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    # Start training
    print("Starting incremental pretraining...")
    print(f"Training on {len(train_dataset)} examples")
    print(f"Validating on {len(val_dataset)} examples")
    
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model
    final_output_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_output_dir, exist_ok=True)
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    metrics["eval_samples"] = len(val_dataset)
    
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Evaluate on test set if available
    if len(dataset_splits['test']) > 0:
        test_dataset = DNADataset(
            dataset_splits['test'], 
            tokenizer, 
            max_length=max_seq_length,
            is_training=False
        )
        test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
        print(f"Test results: {test_results}")
        trainer.save_metrics("test", test_results)
    
    return model, tokenizer, trainer


# Example usage with pre-split dataset
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Incremental Pretraining for MegaDNA')
    parser.add_argument('--model_path', type=str, default="",
                      help='Path to pretrained model')
    parser.add_argument('--tokenizer_path', type=str, default="/hpcfs/fhome/yangchh/BGC-SM/BGCLM-Benchmark/megaBGC/vocab.json",
                      help='Path to tokenizer')
    parser.add_argument('--data_dir', type=str, 
                      default="/hpcfs/fhome/yangchh/BGC-SM/BGCLM-Benchmark/data/BGC",
                      help='Directory containing pre-split train/validation/test .npy files')
    parser.add_argument('--output_dir', type=str, default="./pretraining",
                      help='Output directory for trained model')
    parser.add_argument('--max_seq_length', type=int, default=65536,
                      help='Maximum sequence length')
    parser.add_argument('--train_batch_size', type=int, default=1,
                      help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                      help='Evaluation batch size')
    parser.add_argument('--num_epochs', type=int, default=10000,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                      help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Load pre-split DNA sequences
    dataset_splits = load_presplit_data(args.data_dir)
    
    if len(dataset_splits['train']) < 10:
        raise ValueError("Not enough valid training DNA sequences found. Please check your data directory.")
    
    print(f"\nDataset summary:")
    print(f"  Training: {len(dataset_splits['train'])} sequences")
    print(f"  Validation: {len(dataset_splits['validation'])} sequences")
    print(f"  Test: {len(dataset_splits['test'])} sequences")
    
    # Run incremental pretraining with pre-split data
    model, tokenizer, trainer = setup_incremental_pretraining(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        dataset_splits=dataset_splits,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=8,
        max_seq_length=args.max_seq_length,
        device=args.device,
        logging_steps=50,
        save_steps=200,
        eval_steps=200,
        resume_from_checkpoint=args.resume_from_checkpoint,
        seed=42
    )
    
    print(f"\nTraining completed! Final model saved to: {os.path.join(args.output_dir, 'final_model')}")