from typing import List
from sklearn.model_selection import train_test_split  # 需要安装scikit-learn
import os
import os
import pandas as pd
from typing import List, Optional

# Split dataset function
def split_dataset(sequences: List[str], test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42):
    """
    Split sequences into training, validation, and test sets
    """
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1.0")
    
    # First split: separate test set
    train_val_sequences, test_sequences = train_test_split(
        sequences, test_size=test_size, random_state=random_state
    )

    # Second split: separate validation set from remaining data
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size relative to remaining data
    train_sequences, val_sequences = train_test_split(
        train_val_sequences, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"Dataset split completed:")
    print(f"  Training set: {len(train_sequences)} sequences ({len(train_sequences)/len(sequences):.1%})")
    print(f"  Validation set: {len(val_sequences)} sequences ({len(val_sequences)/len(sequences):.1%})")
    print(f"  Test set: {len(test_sequences)} sequences ({len(test_sequences)/len(sequences):.1%})")
    
    return {
        'train': train_sequences,
        'validation': val_sequences,
        'test': test_sequences
    }

# Load DNA sequences from file (FASTA format support)
def load_dna_sequences(file_path: str, max_sequences: int = None) -> List[str]:
    """
    Load DNA sequences from FASTA file or plain text file
    """
    sequences = []
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Check if it's a FASTA file
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
        # Plain text file, one sequence per line
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('>'):  # Skip FASTA headers if present
                    sequences.append(line)
                    if max_sequences and len(sequences) >= max_sequences:
                        break
                    
    return sequences




import pandas as pd

def fasta_to_bgc_dataframe(fasta_path: str, max_sequences: int = None) -> pd.DataFrame:
    """
    Parse a FASTA file with headers in the format:
    >bgcid|accession|organism|product_type|length_bp (e.g., '10789 bp')|gene_count
    
    Returns a DataFrame with columns:
    ['bgcid', 'accession', 'organism', 'product_type', 'length_bp', 'gene_count', 'sequence']
    
    Args:
        fasta_path (str): Path to the FASTA file.
        max_sequences (int, optional): Maximum number of records to load.
    
    Returns:
        pd.DataFrame
    """
    records = []
    current_header = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                # Save previous record
                if current_header is not None:
                    records.append((current_header, ''.join(current_seq)))
                    if max_sequences and len(records) >= max_sequences:
                        break
                current_header = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)
        
        # Save last record
        if current_header is not None and (not max_sequences or len(records) < max_sequences):
            records.append((current_header, ''.join(current_seq)))

    # Parse each header into structured fields
    parsed = []
    for header, seq in records:
        fields = header.split('|')
        if len(fields) < 6:
            raise ValueError(f"Header has fewer than 6 fields: {header}")
        
        bgcid, accession, organism, product_type, length_str, gene_count_str = fields[:6]
        
        # Clean and convert
        # Remove ' bp' from length_str
        if length_str.endswith(' bp'):
            length_bp = int(length_str[:-3])
        else:
            length_bp = int(length_str)  # fallback
        
        gene_count = int(gene_count_str)

        parsed.append({
            'bgcid': bgcid,
            'accession': accession,
            'organism': organism,
            'product_type': product_type,
            'length_bp': length_bp,
            'gene_count': gene_count,
            'sequence': seq
        })

    return pd.DataFrame(parsed)