import re
import os
import sys
import fire
import tqdm
import torch
import pickle
from pathlib import Path
from itertools import product
import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer
from circuits_benchmark.utils.get_cases import get_cases


# load model
ROOT = Path("../circuits-benchmark/results/var_ll_models")

CFG_PATH = "ll_model_cfg_510.pkl"


def patch_head(dst, hook, src, index):
    """Patching the output of an attention head before the final OV
    computation. `dst` has dimension (batch, sq_len, nhead, d_head)
    `src` has the same dimension. And hook is just a hook point.
    """
    dst[index] = src[index]
    return dst


def setup(case_id, model_id, batch_size):
    cases = get_cases()
    case = [c for c in cases if c.get_name() == str(case_id)][0]

    case_model_dir = ROOT / f"{case_id}-{model_id}/ll_models/{case_id}"
    if not os.path.exists(case_model_dir):
        sys.exit(1)

    tf = HookedTransformer(pickle.load(open(case_model_dir / CFG_PATH, "rb")))

    model_path = case_model_dir / "ll_model_510.pth"

    tf.load_state_dict(torch.load(model_path))

    # get the clean data (exactly 200 samples)
    clean_data = case.get_clean_data(min_samples=200, max_samples=200)
    print("Caching case", case_id, case.__class__.__name__)
    print("Probing", len(clean_data), "examples.")

    loader = clean_data.make_loader(batch_size=batch_size)
    return case, tf, clean_data, loader


def get_reprs(model, loader):
    resids = None
    for x, _ in tqdm.tqdm(loader):
        _, cache = model.run_with_cache(x)
        resid = cache.accumulated_resid(incl_mid=True)

        # only get the odd indices
        resid = resid[1::2,...]

        if resids is None:
            resids = resid
        else:
            resids = torch.cat([resids, resid], dim=1)
        del cache

    return resids.permute(1, 0, 2, 3).flatten(start_dim=1)


def main(case_id, model_id, out_name, batch_size=64, intervene=False, n_nodes=1, n_samples=1, in_out=True):
    """Generates and stores representations under various types of interventions.
    This acts as the first step to computing the alignment between any two models.

    Params:
        case_id (int)
        model_id (int): the seed used to generate the model
        out_name (str): filename to store the representations
        intervene (bool): whether to retrieve representations under
            interventions of attention heads. If this is true, then
            additional files will be stored that denote which attention
            heads were intervened on during the course of representation
            retrieval.
        n_nodes (int): the number of nodes to intervene on.
        n_samples (int): the actual nodes are sampled uniformly at random. So, this
            determines the number of times we are sampling intervention 
        in_out (bool): determines which nodes are being intervened on, those
            within the circuit, or those outside of the circuit. If this is true,
            then we are sampling n_nodes from within the circuit to intervene on.
    """
    case, tf, clean_data, loader = setup(case_id, model_id, batch_size)

    if not intervene:
        reprs = get_reprs(tf, loader)
        pickle.dump(reprs, open(out_name, "wb"))
        return

    # get all possible attn_head hooks that are a part of the model
    act_names = list(filter(lambda x: "attn.hook_result" in x, tf.hook_dict.keys()))



    correspondence = case.get_correspondence()
    n_heads = tf.cfg.n_heads
    head_idx = Index
    head_idx = list(product(range(n_layers), range(n_heads)))
    


if __name__ == "__main__":
    fire.Fire(main)
