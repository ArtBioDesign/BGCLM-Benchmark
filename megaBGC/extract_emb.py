import torch
import numpy as np
from typing import List, Dict, Optional
from functools import reduce
import operator
from megadna_pretrain import DNATokenizer, MegaDNAConfig, MegaDNACausalLM, DNADataset
from utils import split_dataset, load_dna_sequences, fasta_to_bgc_dataframe
from tqdm import tqdm
import os
import pandas as pd


def extract_embeddings(
    model: MegaDNACausalLM,
    input_sequences: List[str],
    tokenizer: DNATokenizer,
    layer_index: int = 0,
    batch_size: int = 8
) -> np.ndarray:
    """Extract embeddings from MEGADNA model with efficient batch processing."""
    if not input_sequences:
        return np.array([])
    
    model.eval()
    device = next(model.parameters()).device
    
    # Calculate total max sequence length from config
    max_seq_len_total = reduce(operator.mul, model.config.max_seq_len)
    
    # Create dataset and dataloader with default collate
    dataset = DNADataset(input_sequences, tokenizer, max_length=max_seq_len_total, is_training=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory='cuda' in str(device)
    )
    
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings", total=len(dataloader)):
            input_ids = batch['input_ids'].to(device)
            
            # Get hidden states from specified layer
            hidden_states = model(input_ids, return_value='embedding').hidden_states
            layer_embeddings = hidden_states[layer_index]  # [batch, seq_len, dim]

            mean_embedding = layer_embeddings.mean(dim=1)

            # mean_embedding = mean_embedding.mean(dim=0, keepdim=True)

            all_embeddings.append( mean_embedding.cpu().numpy() )
    
    return  np.concatenate(all_embeddings, axis=0)

def read_fasta_to_list(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq:  # 如果已经读取了一条序列，就保存它
                    sequences.append(seq)
                    seq = ''
            else:
                seq += line
        if seq:  # 添加最后一条序列
            sequences.append(seq)
    return sequences

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "/hpcfs/fhome/yangchh/genome_lms/megaDNA/test/checkpoint-995200"
    tokenizer = DNATokenizer.from_pretrained(model_path)
    config = MegaDNAConfig.from_pretrained(model_path)

    config.dim = tuple(config.dim)
    config.depth = tuple(config.depth)
    config.max_seq_len = tuple(config.max_seq_len)
    model = MegaDNACausalLM.from_pretrained(model_path, config=config)
    model.to(device) 

    # data_file = '/hpcfs/fhome/yangchh/genome_lms/megaDNA/data/filtered_65536_clean.fasta'
    # dna_sequences = load_dna_sequences(data_file, max_sequences=20203)
    # dna_sequences_df = fasta_to_bgc_dataframe(data_file, max_sequences=20203)
    
    dna_sequences_df = pd.read_csv("/hpcfs/fhome/yangchh/genome_lms/megaDNA/data/mibig_3.1.csv")

    # dataset_splits = split_dataset(
    #     list( dna_sequences_df["sequence"] ),
    #     test_size=0.01, 
    #     val_size=0.1, 
    #     random_state=42
    # )
    
    # Extract embeddings
    embeddings = extract_embeddings(
        model=model,
        input_sequences=  list(dna_sequences_df['sequence']),
        tokenizer=tokenizer,
        layer_index=0,
        batch_size=1
    )   

    embeddings_dir = "/hpcfs/fhome/yangchh/genome_lms/megaDNA/embeddings"
    embeddings_path = os.path.join(embeddings_dir, "mibig_3.1.npy")

    np.save(embeddings_path, embeddings)

    # import pandas as pd
    # validation_df = pd.DataFrame(columns=["sequence"], data=dataset_splits['validation'])
    # sequence_to_bgcid = dna_sequences_df.drop_duplicates('sequence').set_index('sequence')['bgcid'].to_dict()
    # sequence_to_product_type = dna_sequences_df.drop_duplicates('sequence').set_index('sequence')['product_type'].to_dict()
    # validation_df['bgcid'] = validation_df['sequence'].map(sequence_to_bgcid)
    # validation_df['product_type'] = validation_df['sequence'].map(sequence_to_product_type)
    # validation_df_path =  os.path.join(embeddings_dir, "embeddings_validation.df")
    # validation_df[["bgcid", "product_type"]].to_csv(validation_df_path, index=False )

if __name__ == "__main__":
    main()

















































# import torch
# import torch.nn.functional as F
# from typing import List, Literal, Tuple, Optional
# from megadna_pretrain import DNATokenizer, MegaDNAConfig, MegaDNACausalLM

# def extract_embeddings(model, input_sequences, tokenizer, layer_index=0):
#     """
#     Extract embeddings from MEGADNA model
    
#     Args:
#         model: Trained MEGADNA model
#         input_sequences: List of DNA sequences or single sequence string
#         tokenizer: DNATokenizer instance
#         return_layers: 'all' to return all layer embeddings, or specific layer indices
    
#     Returns:
#         Dictionary containing embeddings from different layers
#     """
#     model.eval()  # Set model to evaluation mode
    
#     # Ensure input is a list
#     if isinstance(input_sequences, str):
#         input_sequences = [input_sequences]
    
#     with torch.no_grad():
#         for i, sequence in enumerate(input_sequences):
#             # Tokenize the sequence
#             encoded = tokenizer(sequence, return_tensors='pt')
#             input_ids = encoded['input_ids']

#             eos_token_id = tokenizer.eos_token_id
#             eos_tensor = torch.tensor([[eos_token_id]])
#             input_ids = torch.cat([input_ids, eos_tensor], dim=1)
            
#             # Move to model's device
#             input_ids = input_ids.to(next(model.parameters()).device)
            
#             # Forward pass to get embeddings
#             hidden_states = model(input_ids, return_value='embedding')

#             layer_embeddings = hidden_states[layer_index]

#             mean_embedding = layer_embeddings.mean(dim=1)

#     return mean_embedding.squeeze(0).cpu().numpy() 



# def main():

#     model_path = "/hpcfs/fhome/yangchh/genome_lms/megaDNA/test/checkpoint-995200"
#     tokenizer = DNATokenizer.from_pretrained(model_path)
#     config = MegaDNAConfig.from_pretrained(model_path)

#     config.dim = tuple(config.dim)
#     config.depth = tuple(config.depth)
#     config.max_seq_len = tuple(config.max_seq_len)
#     model = MegaDNACausalLM.from_pretrained(model_path, config=config)

    
#     sample_sequences = [
#         "ATCGATCGATCG",
#         "GGCCAAATTTCG",
#         "ACGTACGTACGT"
#     ]
    
#     embedding = extract_embeddings(model, sample_sequences, tokenizer)
    
#     embeddings_array = np.array(embeddings)




# # Example usage
# if __name__ == "__main__":
    
#     main()


























































































# import torch
# from torch.nn.utils.rnn import pad_sequence
# import torch.nn.functional as F
# from typing import List, Dict, Optional
# from functools import reduce
# import operator
# import numpy as np
# from megadna_pretrain import DNATokenizer, MegaDNAConfig, MegaDNACausalLM, DNADataset


# def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:

#     input_ids_list = [[i for i in item['input_ids'] if i != 0] for item in batch]

#     return {
#         'input_ids': input_ids_list
#     }

# def extract_embeddings(
#     model: MegaDNACausalLM,
#     input_sequences: List[str],
#     tokenizer: DNATokenizer,
#     layer_index: int = 0,
#     batch_size: int = 8,
#     pooling_strategy: str = 'mean'
# ) -> np.ndarray:
#     """
#     Extract embeddings from MEGADNA model with batch processing and proper padding/EOS handling.
    
#     Args:
#         model: Trained MEGADNA model
#         input_sequences: List of DNA sequences
#         tokenizer: DNATokenizer instance
#         layer_index: Index of the layer/stage to extract embeddings from
#         batch_size: Batch size for processing
#         pooling_strategy: 'mean' (default, masked average) or 'eos' (embedding at EOS position)
    
#     Returns:
#         Numpy array of mean-pooled embeddings [num_sequences, hidden_dim]
#     """
#     if not input_sequences:
#         return np.array([])
    
#     model.eval()
#     device = next(model.parameters()).device
    
#     # Compute total max sequence length (product of max_seq_len tuple)
#     max_seq_len_total = reduce(operator.mul, model.config.max_seq_len)
    
#     # Create dataset and dataloader
#     dataset = DNADataset(
#         input_sequences, 
#         tokenizer, 
#         max_length=max_seq_len_total,
#         is_training=False
#         )
    
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=collate_fn,
#         num_workers=0,  # Increase if needed for faster loading
#         pin_memory=True if 'cuda' in str(device) else False
#     )
    
   
    
#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids = batch['input_ids']
            
#             # Forward pass
#             hidden_states = model(input_ids, return_value='embedding')
            
#             layer_embeddings = hidden_states[layer_index]  # [batch, seq_len, dim]

#             mean_embedding = layer_embeddings.mean(dim=1)
            
#     return mean_embedding

# def main():
#     model_path = "/hpcfs/fhome/yangchh/genome_lms/megaDNA/test/checkpoint-995200"
#     tokenizer = DNATokenizer.from_pretrained(model_path)
#     config = MegaDNAConfig.from_pretrained(model_path)

#     config.dim = tuple(config.dim)
#     config.depth = tuple(config.depth)
#     config.max_seq_len = tuple(config.max_seq_len)
#     model = MegaDNACausalLM.from_pretrained(model_path, config=config)
    
#     sample_sequences = [
#         "ATCGATCGATCG",
#         "GGCCAAATTTCG",
#         "ACGTACGTACGT"
#     ]
    
#     # Extract with batching (even for small batch, scalable)
#     embeddings_array = extract_embeddings(
#         model=model,
#         input_sequences=sample_sequences,
#         tokenizer=tokenizer,
#         layer_index=0,
#         batch_size=2,  # Adjustable
#         pooling_strategy='mean'
#     )
    
#     print("Embeddings shape:", embeddings_array.shape)
#     print("First embedding (first 5 dims):", embeddings_array[0][:5])

# if __name__ == "__main__":
#     main()
































































#  import torch
# import torch.nn.functional as F
# from typing import List, Literal, Tuple, Optional
# from megadna_pretrain import DNATokenizer, MegaDNAConfig, MegaDNACausalLM
# import numpy as np

# def extract_embeddings(model, input_sequences, tokenizer, layer_index=0):
#     """
#     Extract embeddings from MEGADNA model
    
#     Args:
#         model: Trained MEGADNA model
#         input_sequences: List of DNA sequences or single sequence string
#         tokenizer: DNATokenizer instance
#         layer_index: Index of the layer from which to extract embeddings
    
#     Returns:
#         Dictionary containing mean embeddings from the specified layer as a numpy array
#     """
#     model.eval()  # Set model to evaluation mode
    
#     # Ensure input is a list
#     if isinstance(input_sequences, str):
#         input_sequences = [input_sequences]
    
#     # Compute total max sequence length
#     from functools import reduce
#     max_seq_len_total = reduce(lambda x, y: x * y, model.config.max_seq_len)
    
#     with torch.no_grad():
#         # Batch tokenize
#         encoded = tokenizer(
#             input_sequences,
#             return_tensors='pt',
#             max_length=max_seq_len_total - 1  # Reserve space for potential EOS if needed
#         )
#         input_ids = encoded['input_ids']
        
        
#         #不够最大长度，进行补全
        
#         # Optionally add EOS to each sequence if desired (to match training)
#         eos = torch.full((input_ids.shape[0], 1), tokenizer.eos_token_id, dtype=input_ids.dtype, device=input_ids.device)

#         input_ids = torch.cat([input_ids, eos], dim=1)

#         # Move to model's device
#         device = next(model.parameters()).device
#         input_ids = input_ids.to(device)
        
#         # Forward pass to get embeddings
#         outputs = model(input_ids, return_value='embedding')
#         hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else outputs
        
#         layer_embeddings: torch.Tensor = hidden_states[layer_index]
        
#         # Mean pool over sequence dimension (assuming shape: batch, seq, dim)
#         mean_embeddings: torch.Tensor = layer_embeddings.mean(dim=1)
        
#     return {
#         f'layer_{layer_index}_mean_embedding': mean_embeddings.cpu().numpy()
#     }

# def main():
#     model_path = "/hpcfs/fhome/yangchh/genome_lms/megaDNA/test/checkpoint-995200"
#     tokenizer = DNATokenizer.from_pretrained(model_path)
#     config = MegaDNAConfig.from_pretrained(model_path)

#     config.dim = tuple(config.dim)
#     config.depth = tuple(config.depth)
#     config.max_seq_len = tuple(config.max_seq_len)
#     model = MegaDNACausalLM.from_pretrained(model_path, config=config)

    
#     sample_sequences = [
#         "ATCGATCGATCG",
#         "GGCCAAATTTCG",
#         "ACGTACGTACGT"
#     ]
    
#     embeddings = extract_embeddings(model, sample_sequences, tokenizer)
    
#     embeddings_array = embeddings['layer_0_mean_embedding']


# # Example usage
# if __name__ == "__main__":
    
#     main()























































































# import torch
# import numpy as np
# from typing import List, Union, Dict, Optional, Tuple
# from megadna_pretrain import DNATokenizer, MegaDNAConfig, MegaDNACausalLM, DNADataset

# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm

# def extract_embeddings(
#     model: MegaDNACausalLM,
#     input_sequences: Union[str, List[str]],
#     tokenizer: DNATokenizer,
#     layer_indices: Optional[Union[int, List[int]]] = 0,
#     batch_size: int = 8,
#     max_length: int = 1024,
#     pooling_strategy: str = 'mean',
#     device: Optional[torch.device] = None,
#     layer_idx: int = 0
# ) -> Union[Dict[int, np.ndarray], np.ndarray]:
#     """
#     Extract embeddings from MEGADNA model efficiently with batch processing
    
#     Args:
#         model: Trained MEGADNA model
#         input_sequences: List of DNA sequences or single sequence string
#         tokenizer: DNATokenizer instance
#         layer_indices: Single layer index or list of layer indices to extract. 
#                       None to return all layers.
#         batch_size: Batch size for processing sequences
#         max_length: Maximum sequence length for tokenization
#         pooling_strategy: 'mean', 'cls', or 'eos' for pooling strategy
#         device: Device to run the model on (auto-detected if None)
    
#     Returns:
#         If layer_indices is a single int: numpy array of shape [num_sequences, hidden_dim]
#         If layer_indices is a list or None: dictionary mapping layer index to embeddings array
#     """
#     model.eval()
#     if device is None:
#         device = next(model.parameters()).device
    
#     # Convert single sequence to list
#     if isinstance(input_sequences, str):
#         input_sequences = [input_sequences]
    
#     # Create dataset and dataloader
#     dataset = DNADataset(input_sequences, tokenizer, max_length)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
#     # # Determine which layers to extract
#     # all_layer_embeddings = {}
#     # if layer_indices is None:
#     #     # Get total number of layers from model config
#     #     num_layers = sum(model.config.depth)  # Total layers across all stages
#     #     layer_indices = list(range(num_layers))
#     # elif isinstance(layer_indices, int):
#     #     layer_indices = [layer_indices]
    
#     # for layer_idx in layer_indices:
#     #     all_layer_embeddings[layer_idx] = []
    
#     # Process batches
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Extracting embeddings"):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             seq_lengths = batch['sequence_length'].cpu().numpy()
            
#             # Get hidden states for all layers
#             hidden_states = model(input_ids, return_value='embedding')
            
#             # Process each requested layer
#             # for layer_idx in layer_indices:
#             #     if layer_idx >= len(hidden_states):
#             #         raise ValueError(f"Layer index {layer_idx} exceeds available layers ({len(hidden_states)})")
                
#             layer_hidden = hidden_states[layer_idx]  # Shape: [batch, seq_len, hidden_dim]
                
#             # Apply pooling strategy
#             if pooling_strategy == 'mean':
#                 # Mask out padding tokens before averaging
#                 expanded_mask = attention_mask.unsqueeze(-1).float()
#                 masked_hidden = layer_hidden * expanded_mask
#                 summed = masked_hidden.sum(dim=1)
#                 counts = expanded_mask.sum(dim=1).clamp(min=1)
#                 pooled = summed / counts
#             elif pooling_strategy == 'cls' or pooling_strategy == 'eos':
#                 # Use the last non-padding token (EOS token)
#                 batch_indices = torch.arange(layer_hidden.size(0))
#                 token_indices = torch.tensor([min(l-1, layer_hidden.size(1)-1) for l in seq_lengths])
#                 pooled = layer_hidden[batch_indices, token_indices]
#             else:
#                 raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
                
#             # all_layer_embeddings[layer_idx].append(pooled.cpu().numpy())
    
#     # Concatenate results for each layer
#     for layer_idx in all_layer_embeddings:
#         all_layer_embeddings[layer_idx] = np.concatenate(all_layer_embeddings[layer_idx], axis=0)
    
#     # Return appropriate format
#     if len(layer_indices) == 1:
#         return all_layer_embeddings[layer_indices[0]]
#     return all_layer_embeddings

# def main():
#     model_path = "/hpcfs/fhome/yangchh/genome_lms/megaDNA/test/checkpoint-995200"
    
#     # Load tokenizer and model with proper configuration
#     tokenizer = DNATokenizer.from_pretrained(model_path)
#     config = MegaDNAConfig.from_pretrained(model_path)
    
#     # Ensure tuples are properly converted
#     config.dim = tuple(config.dim)
#     config.depth = tuple(config.depth)
#     config.max_seq_len = tuple(config.max_seq_len)
    
#     model = MegaDNACausalLM.from_pretrained(model_path, config=config)
#     model.eval()
    
#     # Sample sequences
#     sample_sequences = [
#         "ATCGATCGATCG",
#         "GGCCAAATTTCG", 
#         "ACGTACGTACGT"
#     ]
    
#     # Extract embeddings with batch processing
#     embeddings = extract_embeddings(
#         model=model,
#         input_sequences=sample_sequences,
#         tokenizer=tokenizer,
#         layer_indices=0,  # Extract from first layer
#         batch_size=2,     # Process 2 sequences at a time
#         max_length=128,   # Maximum sequence length
#         pooling_strategy='mean',
#         device='cuda' if torch.cuda.is_available() else 'cpu'
#     )
    
#     print("Embeddings shape:", embeddings.shape)
#     print("Embedding type:", type(embeddings))
#     print("First embedding:", embeddings[0][:10])  # Show first 10 dimensions
    


# if __name__ == "__main__":
#     main()





























































