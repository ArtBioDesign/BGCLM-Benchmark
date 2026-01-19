"""
HyenaDNA Configuration for HuggingFace Transformers
"""
from transformers import PretrainedConfig


class HyenaDNAConfig(PretrainedConfig):
    """
    Configuration class for HyenaDNA model.
    
    Args:
        d_model (int): Model dimension (hidden size). Default: 256
        n_layer (int): Number of HyenaOperator layers. Default: 4
        d_inner (int): Hidden dimension of FFN. Default: 1024 (4 * d_model)
        vocab_size (int): Vocabulary size. Default: 12 (7 special tokens + 5 DNA chars)
        l_max (int): Maximum sequence length. Default: 1024
        order (int): Hyena operator order. Default: 2
        filter_order (int): Filter order for Hyena. Default: 64
        resid_dropout (float): Residual dropout rate. Default: 0.0
        embed_dropout (float): Embedding dropout rate. Default: 0.1
        layer_norm_epsilon (float): LayerNorm epsilon. Default: 1e-5
        use_flash_attn (bool): Whether to use Flash Attention when available. Default: True
        pad_token_id (int): Padding token ID. Default: 4
        bos_token_id (int): Beginning of sequence token ID. Default: 2
        eos_token_id (int): End of sequence token ID. Default: 1
    """
    model_type = "hyenadna"
    
    def __init__(
        self,
        d_model: int = 256,
        n_layer: int = 4,
        d_inner: int = None,
        vocab_size: int = 12,
        l_max: int = 1024,
        order: int = 2,
        filter_order: int = 64,
        resid_dropout: float = 0.0,
        embed_dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = False,
        use_flash_attn: bool = True,
        pad_token_id: int = 4,
        bos_token_id: int = 2,
        eos_token_id: int = 1,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_inner = d_inner if d_inner is not None else 4 * d_model
        self.vocab_size = vocab_size
        self.l_max = l_max
        self.order = order
        self.filter_order = filter_order
        self.resid_dropout = resid_dropout
        self.embed_dropout = embed_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.residual_in_fp32 = residual_in_fp32
        self.use_flash_attn = use_flash_attn
        
        # Layer configuration for HyenaOperator
        self.layer = {
            'd_model': d_model,
            'l_max': l_max,
            'order': order,
            'filter_order': filter_order,
        }


# Predefined model configurations
HYENADNA_CONFIGS = {
    'hyenadna-tiny-1k': HyenaDNAConfig(
        d_model=128,
        n_layer=2,
        l_max=1024,
    ),
    'hyenadna-small-1k': HyenaDNAConfig(
        d_model=256,
        n_layer=4,
        l_max=1024,
    ),
    'hyenadna-medium-8k': HyenaDNAConfig(
        d_model=256,
        n_layer=8,
        l_max=8192,
    ),
    'hyenadna-large-32k': HyenaDNAConfig(
        d_model=512,
        n_layer=8,
        l_max=32768,
    ),
    'hyenadna-large-160k': HyenaDNAConfig(
        d_model=512,
        n_layer=8,
        l_max=160000,
    ),
}


def get_config(model_name: str) -> HyenaDNAConfig:
    """Get predefined configuration by model name."""
    if model_name in HYENADNA_CONFIGS:
        return HYENADNA_CONFIGS[model_name]
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                        f"Available: {list(HYENADNA_CONFIGS.keys())}")
