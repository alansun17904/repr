import re
import os
import sys
import fire
import tqdm
import torch
import pickle
from pathlib import Path
from transformer_lens import HookedTransformer
from circuits_benchmark.utils.get_cases import get_cases


# load model
ROOT = Path("../circuits-benchmark/results/var_ll_models")

CFG_PATH = "ll_model_cfg_510.pkl"


def main(case_id, model_id, out_name, batch_size=64):

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
    resids = None
    for x, _ in tqdm.tqdm(loader):
        _, cache = tf.run_with_cache(x)
        resid = cache.accumulated_resid()
        if resids is None:
            resids = resid
        else:
            resids = torch.cat([resids, resid], dim=1)
        del cache

    flats = resids.permute(1, 0, 2, 3).flatten(start_dim=1)

    pickle.dump(flats, open(out_name, "wb"))


if __name__ == "__main__":
    fire.Fire(main)
