"""
HyenaDNA DNA Sequence Generation Script

Generate DNA sequences using a trained HyenaDNA model.
"""

import torch
import os
import sys
import argparse
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyenaDNA import HyenaDNACausalLM
from tokenizer import Character_Tokenizer


def load_model_and_tokenizer(model_path: str, device: str = 'cuda'):
    """
    Load trained HyenaDNA model and tokenizer.
    
    Args:
        model_path: Path to model directory
        device: Device to load model on
        
    Returns:
        model, tokenizer
    """
    # Load tokenizer
    tokenizer = Character_Tokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=1024 * 1024,  # Allow very long sequences
        padding_side='right',
    )
    
    # Load model
    model = HyenaDNACausalLM.from_pretrained(model_path, device=device)
    model.eval()
    
    return model, tokenizer


def generate_sequences(
    model: HyenaDNACausalLM,
    tokenizer: Character_Tokenizer,
    prompt: Optional[str] = None,
    num_sequences: int = 1,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    do_sample: bool = True,
    device: str = 'cuda',
) -> List[str]:
    """
    Generate DNA sequences.
    
    Args:
        model: Trained HyenaDNA model
        tokenizer: Character tokenizer
        prompt: Optional prompt sequence to start generation
        num_sequences: Number of sequences to generate
        max_length: Maximum sequence length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        do_sample: Whether to sample or use greedy decoding
        device: Device to use
        
    Returns:
        List of generated DNA sequences
    """
    model.eval()
    
    # Prepare prompt
    if prompt is not None:
        prompt = prompt.upper()
        encoding = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(device)
        # Repeat for batch
        input_ids = input_ids.repeat(num_sequences, 1)
    else:
        # Start with BOS token
        input_ids = torch.tensor([[tokenizer.bos_token_id]] * num_sequences, device=device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode sequences
    sequences = []
    for ids in generated_ids:
        seq = tokenizer.decode(ids, skip_special_tokens=True)
        sequences.append(seq)
    
    return sequences


def save_sequences_fasta(
    sequences: List[str],
    output_file: str,
    prefix: str = "hyena_generated"
):
    """
    Save sequences to FASTA file.
    
    Args:
        sequences: List of DNA sequences
        output_file: Output file path
        prefix: Sequence ID prefix
    """
    with open(output_file, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">{prefix}_{i+1} length={len(seq)}\n")
            # Write sequence in 80-character lines
            for j in range(0, len(seq), 80):
                f.write(seq[j:j+80] + '\n')
    
    print(f"Saved {len(sequences)} sequences to {output_file}")


def validate_sequences(sequences: List[str]) -> dict:
    """
    Validate generated DNA sequences.
    
    Args:
        sequences: List of DNA sequences
        
    Returns:
        Validation statistics
    """
    valid_chars = set('ACGTN')
    stats = {
        'total': len(sequences),
        'valid': 0,
        'lengths': [],
        'gc_contents': [],
        'n_ratios': [],
    }
    
    for seq in sequences:
        seq_upper = seq.upper()
        
        # Check if all characters are valid
        is_valid = all(c in valid_chars for c in seq_upper)
        if is_valid:
            stats['valid'] += 1
        
        # Calculate length
        stats['lengths'].append(len(seq))
        
        # Calculate GC content
        if len(seq) > 0:
            gc_count = seq_upper.count('G') + seq_upper.count('C')
            gc_content = gc_count / len(seq)
            stats['gc_contents'].append(gc_content)
            
            # Calculate N ratio
            n_count = seq_upper.count('N')
            n_ratio = n_count / len(seq)
            stats['n_ratios'].append(n_ratio)
    
    # Calculate averages
    if stats['lengths']:
        stats['avg_length'] = sum(stats['lengths']) / len(stats['lengths'])
        stats['min_length'] = min(stats['lengths'])
        stats['max_length'] = max(stats['lengths'])
    
    if stats['gc_contents']:
        stats['avg_gc'] = sum(stats['gc_contents']) / len(stats['gc_contents'])
    
    if stats['n_ratios']:
        stats['avg_n_ratio'] = sum(stats['n_ratios']) / len(stats['n_ratios'])
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Generate DNA sequences with HyenaDNA')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--output_file', type=str, default='generated_sequences.fasta',
                        help='Output FASTA file')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Optional prompt sequence')
    parser.add_argument('--num_sequences', type=int, default=10,
                        help='Number of sequences to generate')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum sequence length')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=None,
                        help='Nucleus sampling parameter')
    parser.add_argument('--greedy', action='store_true',
                        help='Use greedy decoding instead of sampling')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed(args.seed)
    
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    
    print(f"\nGenerating {args.num_sequences} sequences...")
    print(f"  Max length: {args.max_length}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Sampling: {'greedy' if args.greedy else 'stochastic'}")
    if args.prompt:
        print(f"  Prompt: {args.prompt[:50]}{'...' if len(args.prompt) > 50 else ''}")
    
    sequences = generate_sequences(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        num_sequences=args.num_sequences,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
        device=args.device,
    )
    
    # Validate sequences
    print("\nValidation statistics:")
    stats = validate_sequences(sequences)
    print(f"  Total sequences: {stats['total']}")
    print(f"  Valid sequences: {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
    print(f"  Avg length: {stats.get('avg_length', 0):.1f}")
    print(f"  Length range: {stats.get('min_length', 0)} - {stats.get('max_length', 0)}")
    print(f"  Avg GC content: {stats.get('avg_gc', 0)*100:.1f}%")
    print(f"  Avg N ratio: {stats.get('avg_n_ratio', 0)*100:.2f}%")
    
    # Save to file
    save_sequences_fasta(sequences, args.output_file)
    
    # Print sample
    print("\nSample generated sequence:")
    sample = sequences[0]
    if len(sample) > 200:
        print(f"  {sample[:100]}...{sample[-100:]}")
    else:
        print(f"  {sample}")


if __name__ == "__main__":
    main()
