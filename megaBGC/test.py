import sys, torch
sys.path.append("/hpcfs/fhome/yangchh/genome_lms/megaDNA")

model_path = "/hpcfs/fhome/yangchh/genome_lms/megaDNA/test/checkpoint-995200"
device = "cuda" if torch.cuda.is_available() else "cpu"

from megadna_pretrain import DNATokenizer, MegaDNAConfig, MegaDNACausalLM

# --- 加载 tokenizer 和模型 ---
tokenizer = DNATokenizer.from_pretrained(model_path)
config = MegaDNAConfig.from_pretrained(model_path)


config.dim = tuple(config.dim)
config.depth = tuple(config.depth)
config.max_seq_len = tuple(config.max_seq_len)
model = MegaDNACausalLM.from_pretrained(model_path, config=config)

model.to(device)
model.eval()


input_ids=[]
test_seq = "ATCG"
for char in test_seq.upper():
    if char in tokenizer.token_to_id:
        input_ids.append(tokenizer.token_to_id[char])

model = model.to(dtype=torch.bfloat16) 


input_tensor = torch.tensor([input_ids], device=device)

output = model(input_tensor, return_value = 'embedding')


generated_ids = model.generate(
                input_ids=input_tensor,
                max_length=65536,
                temperature=0.95,
                filter_thres=0
            )


generated_sequence = tokenizer.decode(generated_ids[0], skip_special_tokens=False)