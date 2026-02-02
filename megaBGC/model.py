from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer
)

# Make sure MEGABYTE_pytorch is in path
from MEGABYTE_pytorch import MEGABYTE
from MEGABYTE_pytorch.megabyte import reduce_mult, remainder_to_mult, default, exists
from MEGABYTE_pytorch.megabyte import pack_one, unpack_one, top_k, gumbel_sample
from itertools import zip_longest
import torch, os
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch.utils.data import Dataset, DataLoader
from transformers.modeling_outputs import CausalLMOutput
from transformers.configuration_utils import PretrainedConfig
from beartype import beartype
from beartype.typing import Tuple, Union
from tqdm import tqdm
from typing import List, Type, Optional, Dict, Any

class MEGADNA(MEGABYTE):

    @beartype
    def __init__(
        self,
        *,
        num_tokens,
        dim: Union[Tuple, int],
        depth: Tuple,
        max_seq_len: Tuple,
        **kwargs
    ):
        super().__init__(
            num_tokens = num_tokens,
            dim = dim,
            depth = depth,
            max_seq_len = max_seq_len,
            dim_head = 64,
            heads = 8,
            attn_dropout = 0.,
            ff_mult = 4,
            ff_dropout = 0.,
            pad_id = 0,
            rel_pos = False,
            pos_emb = False,
            flash_attn = True,
            **kwargs
        )

    def generate(self, prime = None, seq_len = 1024, filter_thres = 0.9, temperature = 1., default_batch_size = 1, eos_token_id = None):
        device = next(self.parameters()).device

        seq = prime if exists(prime) else torch.empty((default_batch_size, 0), dtype = torch.long, device = device)
        batch = seq.shape[0]
        finished = None
        if eos_token_id is not None:
            finished = torch.zeros((batch,), dtype = torch.bool, device = device)
            if seq.numel() > 0:
                finished = (seq == eos_token_id).any(dim = -1)
        
        with torch.no_grad():
            for _ in tqdm(range(seq_len - seq.shape[-1])):
                logits = self.forward(seq, return_value='logits')[:, -1]
                logits = top_k(logits, thres = filter_thres)
                sampled = gumbel_sample(logits, dim = -1, temperature = temperature)
                if eos_token_id is not None:
                    sampled = torch.where(
                        finished,
                        torch.full_like(sampled, eos_token_id),
                        sampled
                    )
                seq = torch.cat((seq, rearrange(sampled, 'b -> b 1')), dim = -1)
                del logits, sampled
                if eos_token_id is not None:
                    finished = finished | (seq[:, -1] == eos_token_id)
                    if torch.all(finished):
                        break

        return seq.reshape(batch, -1)

    def forward(self, ids, return_value = 'loss', labels=None):
        """
        Modified forward to be compatible with Hugging Face Trainer
        """
        if return_value not in ['logits', 'embedding', 'loss']:
            raise ValueError('return_value must be one of "embedding", "logits", or "loss"')
        
        batch = ids.shape[0]

        assert ids.ndim in {2, self.stages + 1}
        flattened_dims = ids.ndim == 2
        ids_orig_ndim = ids.ndim

        if ids.numel() == 0:
            return self.forward_empty(ids.shape[0])

        if flattened_dims:
            seq_len = ids.shape[-1]
            multiple_of = reduce_mult(self.max_seq_len[1:])
            padding = remainder_to_mult(seq_len, multiple_of)
            ids = F.pad(ids, (0, padding), value = self.pad_id)
            ids = ids.reshape(batch, -1, *self.max_seq_len[1:])

        b, *prec_dims, device = *ids.shape, ids.device

        assert prec_dims[0] <= self.max_seq_len[0], 'the first dimension of your axial autoregressive transformer must be less than the first tuple element of max_seq_len (like any autoregressive transformer)'
        assert tuple(prec_dims[1:]) == tuple(self.max_seq_len[1:]), 'all subsequent dimensions must match exactly'

        tokens_at_stages = []
        pos_embs = default(self.pos_embs, (None,))

        for ind, pos_emb, token_emb in zip_longest(range(len(prec_dims)), pos_embs, self.token_embs):
            is_first = ind == 0

            tokens = token_emb(ids)
            if exists(pos_emb):
                positions = pos_emb(torch.arange(tokens.shape[-2], device = device))
                tokens = tokens + positions

            tokens_at_stages.insert(0, tokens)

            if is_first:
                continue

            ids = rearrange(ids, '... m n -> ... (m n)')

        prev_stage_tokens_repr = None
        hidden_states = []

        for stage_start_tokens, stage_tokens, transformer, proj in zip(self.start_tokens, tokens_at_stages, self.transformers, self.to_next_transformer_projections):
            stage_tokens, ps = pack_one(stage_tokens, '* n d')
            stage_start_tokens = repeat(stage_start_tokens, 'f -> b 1 f', b = stage_tokens.shape[0])

            stage_tokens = torch.cat((
                stage_start_tokens,
                stage_tokens,
            ), dim = -2)

            if exists(prev_stage_tokens_repr):
                prev_stage_tokens_repr = F.pad(prev_stage_tokens_repr, (0, 0, 1, 0), value = 0.)
                stage_tokens = stage_tokens + prev_stage_tokens_repr

            attended = transformer(stage_tokens)
            hidden_states.append(attended)

            attended = unpack_one(attended, ps, '* n d')
            prev_stage_tokens_repr = proj(attended[..., :-1, :])

        # if return_value == 'embedding':
        #     return hidden_states
            
        logits = self.to_logits(attended)
        start_tokens = logits[(slice(None), *((0,) * (logits.ndim - 2)), slice(None))]   #序列最开始那个位置的logits
        start_tokens = rearrange(start_tokens, 'b d -> b 1 d')   #将start_tokens的维度从(b,d)变成(b,1,d)
        logits = logits[..., 1:, :]     #去掉序列最开始那个位置的logits

        if return_value == 'logits':
            if flattened_dims:
                logits = rearrange(logits, 'b ... c -> b (...) c')
                logits = logits[:, :seq_len]
            return logits
            
        logits = rearrange(logits, 'b ... c -> b (...) c')      #将logits的维度从(b,...,c)变成(b,...,c)
        logits = torch.cat((start_tokens, logits), dim = -2)    #将start_tokens和logits拼接在一起

        # For Hugging Face compatibility, handle labels if provided
        loss = None
        if labels is not None:
            preds = rearrange(logits, 'b n c -> b c n')     #将logits的维度从(b,...,c)变成(b,c,...)
            labels = rearrange(labels, 'b ... -> b (...)')   #将labels的维度从(b,...)变成(b,...)
            
            loss = F.cross_entropy(
                preds[..., :-1],
                labels,
                ignore_index = -100   # 0
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states if return_value == 'embedding' else None
        )

class MegaDNAConfig(PretrainedConfig):
    model_type = "megadna"

    def __init__(
        self,
        vocab_size=6,
        dim=(768, 512, 256),
        depth=(6, 4, 2),
        max_seq_len=(512, 4, 4),
        pad_token_id=0,
        eos_token_id=5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

class DNATokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.txt"}
    DEFAULT_TOKENS = ("**", "#")

    def __init__(
        self,
        vocab_file=None,
        pad_token=DEFAULT_TOKENS[0],
        eos_token=DEFAULT_TOKENS[1],
        **kwargs,
    ):
        vocab = None
        if vocab_file and os.path.isfile(vocab_file):
            if str(vocab_file).endswith(".json"):
                import json
                with open(vocab_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    max_id = max(int(v) for v in data.values())
                    vocab = [None] * (max_id + 1)
                    for tok, idx in data.items():
                        vocab[int(idx)] = tok
                    if any(tok is None for tok in vocab):
                        vocab = [tok for tok, _ in sorted(data.items(), key=lambda kv: kv[1])]
                elif isinstance(data, list):
                    vocab = [str(tok) for tok in data]
            else:
                with open(vocab_file, "r", encoding="utf-8") as f:
                    vocab = [line.strip() for line in f if line.strip()]

        if not vocab:
            vocab = [pad_token, "A", "T", "C", "G", eos_token]

        if pad_token not in vocab and "**" in vocab:
            pad_token = "**"
        if eos_token not in vocab and "#" in vocab:
            eos_token = "#"

        self.vocab = vocab
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id_to_token = {idx: tok for idx, tok in enumerate(self.vocab)}
        self.pad_token_id = self.token_to_id.get(pad_token, 0)
        self.eos_token_id = self.token_to_id.get(eos_token, len(self.vocab) - 1)



        super().__init__(
            pad_token=pad_token,
            eos_token=eos_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        return self.token_to_id.copy()

    def _tokenize(self, text: Any, **kwargs: Any) -> list[str]:
        return list(str(text).upper())

    def _convert_token_to_id(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]

    def _convert_id_to_token(self, index: int) -> str:
        if index in self.id_to_token:
            return self.id_to_token[index]

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: str | None = None,
    ) -> tuple[str, ...]:
        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix or "") + self.vocab_files_names["vocab_file"]
        )
        with open(vocab_file, "w", encoding="utf-8") as writer:
            writer.write("\n".join(self.vocab))
        return (vocab_file,)

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        """Convert token IDs back to DNA sequence"""

        tokens = []
        for token_id in token_ids:
            if isinstance(token_id, list):
                tokens.append(self.decode(token_id, skip_special_tokens, clean_up_tokenization_spaces))
            else:
                token = self._convert_id_to_token(int(token_id))
                token = token.content if hasattr(token, 'content') else str(token)

                if skip_special_tokens and token in [self.pad_token, self.eos_token]:
                    continue
                tokens.append(token)
        
        if isinstance(token_ids[0], list):  # batch decoding
            return [''.join(seq_tokens) for seq_tokens in tokens]
        return ''.join(tokens)

class MegaDNAPreTrainedModel(PreTrainedModel):
    config_class = MegaDNAConfig
    base_model_prefix = "megadna"

    def _init_weights(self, module):
        """Initialize weights like in original MEGABYTE implementation"""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

class MegaDNACausalLM(MegaDNAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.megadna = MEGADNA(
            num_tokens=config.vocab_size,
            dim=config.dim,
            depth=config.depth,
            max_seq_len=config.max_seq_len
        )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        return_value: Optional[str] = "loss",
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.megadna(input_ids, return_value=return_value, labels=labels)
        
        if not return_dict:
            return (outputs.loss, outputs.logits) + (None, None)
        
        return CausalLMOutput(
            loss=outputs.loss if return_value=="loss" else None,
            logits=outputs.logits if return_value in ["loss", "logits"] else None,
            hidden_states=outputs.hidden_states if return_value=="embedding" else None,
            attentions=None
        )

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: int = 8192,
        temperature: float = 0.95,
        filter_thres: float = 0,
        eos_token_id: Optional[int] = None,
        **kwargs
    ):
        return self.megadna.generate(
            prime=input_ids,
            seq_len=max_length,
            temperature=temperature,
            filter_thres=filter_thres,
            eos_token_id=eos_token_id,
            default_batch_size= 1                       
        )

class DNADataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=1024, is_training=True):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        sequence = self.sequences[idx]
        # Truncate if too long
        if len(sequence) > self.max_length-1:
            sequence = sequence[:self.max_length-1]
        
        # Tokenize the sequence
        input_ids = []
        for char in sequence:
            if char in self.tokenizer.token_to_id:
                input_ids.append(self.tokenizer.token_to_id[char])
        
        # Append eos at the true end, then pad to max_length
        input_ids.append(self.tokenizer.eos_token_id)
        if len(input_ids) < self.max_length:
            input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()  # For causal LM, labels are same as input_ids
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': (input_ids != self.tokenizer.pad_token_id).long()
        }
        