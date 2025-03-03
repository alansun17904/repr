import torch
from transformer_lens import HookedTransformer
from circuits_benchmark.utils.get_cases import get_cases


# load model 
cfg_path: str = ""
model_path: str = ""
cases = get_cases() 

tf = HookedTransformer(open(cfg_path, "rb"))
tf.load_state_dict(torch.load(model_path))
