import torch
import os
import numpy as np
import pandas as pd
import sys
from Bio import SeqIO

sys.path.append("/hpcfs/fhome/yangchh/genome_lms/megaDNA")
from megaDNA.megadna import MEGADNA

model_path = "/hpcfs/fhome/yangchh/genome_lms/megaDNA/test/checkpoint-995200"
device = "cuda" if torch.cuda.is_available() else "cpu"

from megadna_pretrain import DNATokenizer, MegaDNAConfig, MegaDNACausalLM

def write_fasta_single(seq_id, sequence, output_file):
    """
    将单条序列写入 FASTA 文件。
    
    Parameters:
        seq_id (str): 序列 ID（不能包含空格，否则会被截断）
        sequence (str): DNA/RNA/蛋白质序列
        output_file (str): 输出文件路径
    """
    with open(output_file, 'w') as f:
        f.write(f">{seq_id}\n")
        # 每行写 80 个字符（符合标准 FASTA 格式）
        for i in range(0, len(sequence), 80):
            f.write(sequence[i:i+80] + "\n")

def fasta_to_dataframe(fasta_file):
    records = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        records.append({
            "id": record.id,
            "description": record.description,
            "sequence": str(record.seq)
        })
    return pd.DataFrame(records)


df = fasta_to_dataframe("/hpcfs/fhome/yangchh/genome_lms/megaDNA/data/BGC0002632.fasta")

tokenizer = DNATokenizer.from_pretrained(model_path)
config = MegaDNAConfig.from_pretrained(model_path)

num_samples = 5 # number of sequences to sample per primer
config.dim = tuple(config.dim)
config.depth = tuple(config.depth)
config.max_seq_len = tuple(config.max_seq_len)
model = MegaDNACausalLM.from_pretrained(model_path, config=config)

model.to(device)
model.eval()

input_ids=[]
test_seq = list(df["sequence"])[0][:1000]
for char in test_seq.upper():
    if char in tokenizer.token_to_id:
        input_ids.append(tokenizer.token_to_id[char])

input_tensor = torch.tensor([input_ids], device=device)


for sample_idx in range(num_samples):

    seq_tokenized = model.generate( input_tensor, 
                                    max_length=37874,
                                    temperature=0.95, 
                                    filter_thres=0.0)

    generated_sequence = tokenizer.decode(seq_tokenized.squeeze().cpu().int())

    contigs = generated_sequence.split('#')

    output_file_path = f"generate_sample{sample_idx + 1}.fna"

    output_file_path = os.path.join("/hpcfs/fhome/yangchh/genome_lms/megaDNA/data", output_file_path)

    with open(output_file_path, "w") as file:
        for idx, contig in enumerate(contigs):
            if len(contig) > 0:
                file.write(f">contig_{idx}\n{contig}\n")
