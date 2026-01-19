"""
HyenaDNA Training Script for DNA Sequence Generation

This script trains HyenaDNA model using causal language modeling (next-token prediction)
for DNA sequence generation.
"""

import torch
import os
import sys
from typing import List, Optional
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset

from hyenaDNA import HyenaDNACausalLM
from tokenizer import Character_Tokenizer
from config import HyenaDNAConfig, get_config


class HyenaDNADataset(Dataset):
    """
    Dataset for HyenaDNA training with DNA sequences.
    """
    
    def __init__(
        self,
        sequences: List[str],
        tokenizer: Character_Tokenizer,
        max_length: int = 1024,
        add_special_tokens: bool = False,
    ):
        """
        Initialize dataset.
        
        Args:
            sequences: List of DNA sequences
            tokenizer: Character tokenizer for DNA
            max_length: Maximum sequence length
            add_special_tokens: Whether to add special tokens
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].upper()
        
        # Tokenize sequence
        encoding = self.tokenizer(
            seq,
            truncation=True,
            max_length=self.max_length - 1,  # Reserve space for EOS
            add_special_tokens=False,  # We'll add EOS manually
            return_tensors=None,
            padding=False,
        )
        
        input_ids = encoding['input_ids']
        if isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        
        # Add EOS token at the end for causal LM
        input_ids = self.tokenizer.build_inputs_with_eos(input_ids)
        
        # For causal LM, labels = input_ids (shifted internally)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(input_ids, dtype=torch.long),
        }


class HyenaDNADataCollator:
    """
    Data collator for HyenaDNA that handles variable length sequences.
    """
    
    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
    
    def __call__(self, features):
        # Get max length in batch
        max_len = min(
            max(len(f['input_ids']) for f in features),
            self.max_length
        )
        
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        
        for f in features:
            input_ids = f['input_ids'][:max_len]
            labels = f['labels'][:max_len]
            
            # Calculate padding
            padding_length = max_len - len(input_ids)
            
            # Pad on the right for causal LM
            if padding_length > 0:
                input_ids = torch.cat([
                    input_ids,
                    torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
                ])
                # Use -100 for padded labels (ignored in loss)
                labels = torch.cat([
                    labels,
                    torch.full((padding_length,), -100, dtype=torch.long)
                ])
                attention_mask = torch.cat([
                    torch.ones(max_len - padding_length, dtype=torch.long),
                    torch.zeros(padding_length, dtype=torch.long)
                ])
            else:
                attention_mask = torch.ones(max_len, dtype=torch.long)
            
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)
        
        return {
            'input_ids': torch.stack(batch_input_ids),
            'labels': torch.stack(batch_labels),
            'attention_mask': torch.stack(batch_attention_mask),
        }


def load_dna_sequences(file_path: str, max_sequences: int = None) -> List[str]:
    """
    Load DNA sequences from FASTA file.
    """
    sequences = []
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    is_fasta = file_path.lower().endswith(('.fasta', '.fa', '.fna'))
    
    if is_fasta:
        current_seq = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(''.join(current_seq))
                        current_seq = []
                        if max_sequences and len(sequences) >= max_sequences:
                            break
                else:
                    current_seq.append(line)
            if current_seq and (not max_sequences or len(sequences) < max_sequences):
                sequences.append(''.join(current_seq))
    else:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('>'):
                    sequences.append(line)
                    if max_sequences and len(sequences) >= max_sequences:
                        break
    
    return sequences


def split_dataset(sequences: List[str], test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42):
    """
    Split sequences into train/val/test sets.
    """
    from sklearn.model_selection import train_test_split
    
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1.0")
    
    train_val_sequences, test_sequences = train_test_split(
        sequences, test_size=test_size, random_state=random_state
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    train_sequences, val_sequences = train_test_split(
        train_val_sequences, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_sequences)} ({len(train_sequences)/len(sequences):.1%})")
    print(f"  Val: {len(val_sequences)} ({len(val_sequences)/len(sequences):.1%})")
    print(f"  Test: {len(test_sequences)} ({len(test_sequences)/len(sequences):.1%})")
    
    return {
        'train': train_sequences,
        'validation': val_sequences,
        'test': test_sequences
    }


def train_hyenadna(
    data_file: str,
    output_dir: str,
    model_name: str = None,
    pretrained_path: str = None,
    max_sequences: int = None,
    max_seq_length: int = 1024,
    d_model: int = 256,
    n_layer: int = 4,
    learning_rate: float = 6e-4,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    num_epochs: int = 10,
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.1,
    logging_steps: int = 100,
    save_steps: int = 500,
    eval_steps: int = 500,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    device: str = 'cuda',
    fp16: bool = False,
    bf16: bool = True,
    deepspeed: str = None,  # Path to DeepSpeed config JSON file
    **kwargs
):
    """
    Train HyenaDNA model for DNA sequence generation.
    
    Args:
        data_file: Path to FASTA file with DNA sequences
        output_dir: Output directory for model checkpoints
        model_name: Predefined model name (e.g., 'hyenadna-small-1k')
        pretrained_path: Path to pretrained model weights
        max_sequences: Maximum number of sequences to use
        max_seq_length: Maximum sequence length
        d_model: Model dimension (if not using predefined)
        n_layer: Number of layers (if not using predefined)
        learning_rate: Learning rate
        train_batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        num_epochs: Number of training epochs
        gradient_accumulation_steps: Gradient accumulation steps
        warmup_ratio: Warmup ratio
        weight_decay: Weight decay
        logging_steps: Logging frequency
        save_steps: Save checkpoint frequency
        eval_steps: Evaluation frequency
        test_size: Test set proportion
        val_size: Validation set proportion
        random_state: Random seed
        device: Device to use
        fp16: Use FP16 training
        bf16: Use BF16 training
    """
    
    # Load sequences
    print(f"Loading sequences from {data_file}...")
    sequences = load_dna_sequences(data_file, max_sequences=max_sequences)
    print(f"Loaded {len(sequences)} sequences")
    
    if len(sequences) < 10:
        raise ValueError("Not enough sequences. Need at least 10.")
    
    # Split dataset
    dataset_splits = split_dataset(
        sequences,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    
    # Create tokenizer
    tokenizer = Character_Tokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=max_seq_length,
        padding_side='right',
    )
    
    # Create datasets
    train_dataset = HyenaDNADataset(
        dataset_splits['train'],
        tokenizer,
        max_length=max_seq_length,
    )
    
    val_dataset = HyenaDNADataset(
        dataset_splits['validation'],
        tokenizer,
        max_length=max_seq_length,
    )
    
    # Create data collator
    data_collator = HyenaDNADataCollator(tokenizer, max_length=max_seq_length)
    
    # Create model configuration
    if model_name:
        config = get_config(model_name)
        config.l_max = max_seq_length
    else:
        config = HyenaDNAConfig(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=tokenizer.vocab_size,
            l_max=max_seq_length,
        )
    
    # Create model
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        model = HyenaDNACausalLM.from_pretrained(pretrained_path, device=device)
    else:
        print(f"Creating new model: d_model={config.d_model}, n_layer={config.n_layer}")
        model = HyenaDNACausalLM(config=config)
    
    model = model.to(device)
    
    # Print model info
    n_params = model.get_num_params()
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        save_safetensors=False,  # Disable safetensors due to weight tying
        # Multi-GPU/DeepSpeed settings
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        dataloader_pin_memory=True,
        deepspeed=deepspeed,  # DeepSpeed config path
        **kwargs
    )
    
    # Compute metrics
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        
        # Calculate perplexity
        shift_logits = torch.tensor(logits[..., :-1, :])
        shift_labels = torch.tensor(labels[..., 1:])
        
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        perplexity = torch.exp(loss)
        
        return {
            'perplexity': perplexity.item(),
        }
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    # Train
    print("\nStarting training...")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_result = trainer.train()
    
    # Save final model
    final_output_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_output_dir, exist_ok=True)
    
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    metrics["eval_samples"] = len(val_dataset)
    
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Test evaluation
    if len(dataset_splits['test']) > 0:
        test_dataset = HyenaDNADataset(
            dataset_splits['test'],
            tokenizer,
            max_length=max_seq_length,
        )
        test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
        print(f"\nTest results: {test_results}")
        trainer.save_metrics("test", test_results)
    
    print(f"\nTraining completed! Model saved to: {final_output_dir}")
    
    return model, tokenizer, trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train HyenaDNA for DNA sequence generation')
    
    # Data arguments
    parser.add_argument('--data_file', type=str, default="/hpcfs/fhome/yangchh/genome_lms/megaDNA/data/filtered_65536_clean.fasta",
                        help='Path to FASTA file with DNA sequences')
    parser.add_argument('--output_dir', type=str, default='./hyena_training',
                        help='Output directory for checkpoints')
    parser.add_argument('--max_sequences', type=int, default=None,
                        help='Maximum number of sequences to use')
    parser.add_argument('--max_seq_length', type=int, default=65536,
                        help='Maximum sequence length')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default=None,
                        help='Predefined model name (e.g., hyenadna-small-1k)')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained model')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--n_layer', type=int, default=8,
                        help='Number of layers')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=6e-4,
                        help='Learning rate')
    parser.add_argument('--train_batch_size', type=int, default=1,
                        help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help='Evaluation batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help='Gradient accumulation steps')
    parser.add_argument('--warmup_ratio', type=float, default=0.03,
                        help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay')
    
    # Logging arguments
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Logging frequency')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint frequency')
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='Evaluation frequency')
    
    # Dataset split
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation set proportion')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Device and DeepSpeed
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 training')
    parser.add_argument('--bf16', action='store_true', default=True,
                        help='Use BF16 training')
    parser.add_argument('--deepspeed', type=str, default=None,
                        help='Path to DeepSpeed config JSON file (e.g., ds_config_zero2.json)')
    
    args = parser.parse_args()
    
    # Run training
    train_hyenadna(
        data_file=args.data_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        pretrained_path=args.pretrained_path,
        max_sequences=args.max_sequences,
        max_seq_length=args.max_seq_length,
        d_model=args.d_model,
        n_layer=args.n_layer,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
        device=args.device,
        fp16=args.fp16,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
    )
