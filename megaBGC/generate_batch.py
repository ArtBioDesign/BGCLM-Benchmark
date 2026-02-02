import torch
import os
import numpy as np
import sys
from tqdm import tqdm

# sys.path.append("/hpcfs/fhome/yangchh/genome_lms/megaDNA")
# from megabgc.megadna import MEGADNA
from megabgc_pretrain import DNATokenizer, MegaDNAConfig, MegaDNACausalLM

# ========== Speed Optimizations ==========
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ========== Configuration ==========
MIN_CONTIG_LENGTH = 100  # Minimum length to save to main file
PRIME_LENGTH = 4


def generate_sequences(
    model_path: str,
    output_file: str,
    num_samples: int = 1000,
    batch_size: int = 1,
    max_length: int = 65536,
    device: str = "cuda",
    use_fp16: bool = True,
    compile_model: bool = False,
    temperature: float = 0.95,
    filter_thres: float = 0.0
):
    """
    Generate DNA sequences using MegaDNA model.
    
    Args:
        model_path: Path to pretrained model checkpoint
        output_file: Output FASTA file path for long sequences
        num_samples: Number of sequences to generate
        batch_size: Batch size for generation (reduce if OOM)
        max_length: Maximum sequence length
        device: Device to use (cuda/cpu)
        use_fp16: Use mixed precision for faster inference
        compile_model: Use torch.compile (PyTorch 2.0+)
        temperature: Sampling temperature
        filter_thres: Top-k filtering threshold (0 = no filtering)
    """
    import time
    start_time = time.time()
    
    print(f"Loading model from {model_path}...")
    tokenizer = DNATokenizer.from_pretrained(model_path)
    config = MegaDNAConfig.from_pretrained(model_path)
    
    # Fix tuple config issues
    config.dim = tuple(config.dim)
    config.depth = tuple(config.depth)
    config.max_seq_len = tuple(config.max_seq_len)
    
    model = MegaDNACausalLM.from_pretrained(model_path, config=config)
    model.to(device)
    
    # ========== Mixed Precision ==========
    if use_fp16 and device == "cuda":
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("✓ Using BFloat16 precision")
        else:
            dtype = torch.float16
            print("✓ Using Float16 precision")
        model = model.to(dtype=dtype)
    else:
        dtype = torch.float32
        print("✓ Using Float32 precision")
    
    # ========== Optional: torch.compile ==========
    if compile_model and hasattr(torch, 'compile'):
        print("🔧 Compiling model with torch.compile (this may take 1-3 min)...")
        compile_start = time.time()
        model = torch.compile(model, mode="reduce-overhead")
        print(f"✓ Model compiled in {time.time() - compile_start:.1f}s")
    model.eval()
    
    # ========== Prepare Output Files ==========
    output_dir = os.path.dirname(output_file)
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    short_file = os.path.join(output_dir, f"{base_name}_short.fasta")
    
    print(f"\n{'='*60}")
    print(f"🧬 Generating {num_samples} sequences (max_length={max_length})")
    print(f"{'='*60}")
    print(f"  Batch size: {batch_size}")
    print(f"  Main output (len > {MIN_CONTIG_LENGTH}): {output_file}")
    print(f"  Short sequences output: {short_file}")
    print(f"{'='*60}\n")
    
    # ========== Statistics ==========
    stats = {
        "total_generated": 0,
        "long_contigs": 0,
        "short_contigs": 0,
        "total_contigs": 0,
        "errors": 0
    }
    
    nucleotides = ['**', 'A', 'C', 'G', 'T', '#']
    
    def token2nucleotide(s):
        return nucleotides[s]
    
    # Buffers for batch writing (reduce I/O overhead)
    main_buffer = []
    short_buffer = []
    FLUSH_INTERVAL = 10  # Flush every 10 sequences
    
    with open(output_file, "w") as f_main, open(short_file, "w") as f_short:
        pbar = tqdm(total=num_samples, desc="Generating", unit="seq", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        with torch.no_grad(), torch.amp.autocast(
            device_type='cuda', 
            dtype=dtype if use_fp16 else torch.float32, 
            enabled=(device == "cuda")
        ):
            while stats["total_generated"] < num_samples:
                current_batch_size = min(batch_size, num_samples - stats["total_generated"])
                
                # Generate random primers
                random_primers = np.random.choice(
                    np.arange(1, 5), 
                    size=(current_batch_size, PRIME_LENGTH)
                )
                input_tensor = torch.tensor(random_primers, dtype=torch.long, device=device)
                
                # Less frequent logging to reduce overhead
                if stats["total_generated"] % 50 == 0 and stats["total_generated"] > 0:
                    primer_dna = ''.join(map(token2nucleotide, input_tensor[0].cpu().numpy()))
                    elapsed = time.time() - start_time
                    rate = stats["total_generated"] / elapsed if elapsed > 0 else 0
                    pbar.write(f"[Progress] {stats['total_generated']} seqs | {rate:.2f} seq/s | Primer: {primer_dna}")
                
                try:
                    # Generate sequences
                    seq_tokenized = model.generate(
                        input_tensor,
                        max_length=max_length,
                        temperature=temperature,
                        filter_thres=filter_thres,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    
                    # Decode and save each sequence in the batch
                    for i in range(current_batch_size):
                        seq_ids = seq_tokenized[i].squeeze().cpu().int()
                        generated_sequence = tokenizer.decode(seq_ids)
                        
                        # Split by stop token
                        contigs = generated_sequence.split('#')
                        
                        # ========== Log Sequence Information ==========
                        seq_num = stats['total_generated'] + 1
                        pbar.write(f"\n{'='*70}")
                        pbar.write(f"📝 Sequence #{seq_num}")
                        pbar.write(f"{'='*70}")
                        pbar.write(f"  Total length: {len(generated_sequence):,} bp")
                        pbar.write(f"  Number of contigs: {len([c for c in contigs if c.strip()])}")
                        
                        # Preview full sequence (first and last 50bp)
                        if len(generated_sequence) > 100:
                            preview = f"{generated_sequence[:50]}...{generated_sequence[-50:]}"
                        else:
                            preview = generated_sequence
                        pbar.write(f"  Full sequence preview: {preview}")
                        
                        for part_idx, contig in enumerate(contigs):
                            contig = contig.strip()
                            if not contig:
                                continue
                            
                            stats["total_contigs"] += 1
                            
                            # Create unique sequence ID
                            seq_id = f"gen_sample_{seq_num}"
                            if len(contigs) > 1:
                                seq_id += f"_part{part_idx + 1}"
                            
                            # Log contig details
                            contig_type = "LONG" if len(contig) > MIN_CONTIG_LENGTH else "SHORT"
                            pbar.write(f"\n  📌 Contig {part_idx + 1}/{len([c for c in contigs if c.strip()])} [{contig_type}]")
                            pbar.write(f"     ID: {seq_id}")
                            pbar.write(f"     Length: {len(contig):,} bp")
                            
                            # Show contig sequence preview
                            if len(contig) > 80:
                                contig_preview = f"{contig[:40]}...{contig[-40:]}"
                            else:
                                contig_preview = contig
                            pbar.write(f"     Sequence: {contig_preview}")
                            
                            # Buffer writes instead of immediate flush
                            if len(contig) > MIN_CONTIG_LENGTH:
                                main_buffer.append(f">{seq_id}\n{contig}\n")
                                stats["long_contigs"] += 1
                                pbar.write(f"     ✓ Saved to main file")
                            else:
                                short_buffer.append(f">{seq_id}_len{len(contig)}\n{contig}\n")
                                stats["short_contigs"] += 1
                                pbar.write(f"     ✓ Saved to short file")
                        
                        stats["total_generated"] += 1
                        pbar.update(1)
                        pbar.write("")  # Empty line for readability
                        
                        # Flush buffers periodically
                        if stats["total_generated"] % FLUSH_INTERVAL == 0:
                            if main_buffer:
                                f_main.writelines(main_buffer)
                                f_main.flush()
                                main_buffer = []
                            if short_buffer:
                                f_short.writelines(short_buffer)
                                f_short.flush()
                                short_buffer = []
                        
                        if stats["total_generated"] >= num_samples:
                            break
                            
                except Exception as e:
                    stats["errors"] += 1
                    pbar.write(f"⚠️  Error during generation: {e}")
                    import traceback
                    traceback.print_exc()
                    # Skip this batch and continue
                    stats["total_generated"] += current_batch_size
                    pbar.update(current_batch_size)
                    continue
        
        # Final flush
        if main_buffer:
            f_main.writelines(main_buffer)
        if short_buffer:
            f_short.writelines(short_buffer)
        
        pbar.close()
    
    # ========== Print Summary ==========
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("🎉 Generation Complete!")
    print("=" * 60)
    print(f"  Total sequences generated: {stats['total_generated']}")
    print(f"  Total contigs: {stats['total_contigs']}")
    print(f"  Long contigs (> {MIN_CONTIG_LENGTH} bp): {stats['long_contigs']}")
    print(f"  Short contigs (<= {MIN_CONTIG_LENGTH} bp): {stats['short_contigs']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Average speed: {stats['total_generated']/total_time:.2f} sequences/sec")
    print("=" * 60)
    
    return stats


if __name__ == "__main__":
    # ========== Settings ==========
    MODEL_PATH = "/hpcfs/fhome/yangchh/BGC-SM/BGCLM-Benchmark/megaBGC/pretraining/final_model"
    OUTPUT_FILE = "/hpcfs/fhome/yangchh/BGC-SM/BGCLM-Benchmark/data/generate/megabgc_generated_100_sequences.fasta"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        generate_sequences(
            model_path=MODEL_PATH,
            output_file=OUTPUT_FILE,
            num_samples=100,
            max_length=1000,          # Full BGC length
            batch_size=8,             # RTX 5090 can handle large batches
            device=device,
            use_fp16=True,
            compile_model=True,        # Enable for 15-30% speedup
            temperature=0.95,
            filter_thres=0.5
        )
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
