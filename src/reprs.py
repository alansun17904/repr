import fire
import torch
import pickle
from pathlib import Path
from transformer_lens import HookedTransformer
from circuits_benchmark.utils.get_cases import get_cases


# load model 
ROOT = Path("")

CFG_PATH = ROOT / "ll_model_cfg_510.pkl"


def main(case_id, model_id, out_name, batch_size=256):

    cases = get_cases()
    case = [c for c in cases if c.__class__.__name__[:4] in str(case_id)][0]

    tf = HookedTransformer(pickle.load(open(CFG_PATH, "rb")))

    model_path = ROOT / f"{case_id}-{model_id}" / "ll_model_510.pth"

    tf.load_state_dict(torch.load(model_path))

    # get the clean data
    clean_data = case.get_clean_data(min_samples=200, max_samples=200)
    print("Probing", len(clean_data), "examples.")
    loader = clean_data.make_loader(batch_size=batch_size)
    resids = None
    logits = None
    for x, _ in loader:
        _, cache = tf.run_with_cache(x)
        resid = cache.accumulated_resid()
        if resid is None:
            resid = resid
        else:
            resid = torch.cat([resids, resid], dim=1)
        del cache
    
    flats = resids.permute(1, 0, 2, 3).flatten(start_dim=1)

    pickle.dump(flats, open(out_name, "wb"))


if __name__ == "__main__":
    fire.Fire(main)