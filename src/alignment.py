import re
import sys
import fire
import torch
import pickle
import itertools
import numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
from transformer_lens import HookedTransformer
from circuits_benchmark.utils.get_cases import get_cases


CFG_PATH = "ll_model_cfg_510.pkl" 


def ridge_fit(X, Y):
    rcv = RidgeCV(alpha_per_target=True, fit_intercept=False)
    rcv.fit(X, Y)
    return rcv.score(X, Y)


def patch_head(dst, hook, src, head_index):
    """Patching the output of an attention head before the final OV
    computation. `dst` has dimension (batch, sq_len, nhead, d_head)
    `src` has the same dimension. And hook is just a hook point.
    """
    dst[:, :, head_inex, :] = src[:, :, head_inex, :]
    return dst


def main(case_id, model_id, out_name, batch_size=256, intervene=False):

    if not intervene:
        
        ROOT = Path("data/reprs/")
        cases = list(set([
            re.search(r"\d+", f_name) for f_name in os.listdir()
        ]))

        # compare all of the implementaitons within the same class
        same_cls_corr = dict()
        for c in case:
            same_case = list(filter(lambda x: x.startswith(f"case_{c}"), os.listdir()))
            run_corrs = [] 
            tot = 0
            for pair in itertools.combinations(same_case, 2):
                c1_fname, c2_fname = pair
                c1, c2 = (
                    pickle.load(open(c1_fname, "rb")),
                    pickle.load(open(c2_fname, "rb"))
                )
                tot += 1
                run_corrs.append(ridge_fit(c1, c2))

            same_cls_corr[int(c)] = sum(run_corrs) / tot

        pickle.dump(same_cls_corr, open(f"{out_name}_same.pkl", "Wb"))
        # compare alignment across different tasks
        diff_cases = dict()
        for c in case:
            same_case = list(filter(lambda x: x.startswith(f"case_{c}"), os.listdir()))
            diff_case = list(filter(lambda x: not x.startswith(f"case_{c}"), os.listdir()))
            tot = 0
            run_corrs = []
            cross_generator = itertools.product(same_case, diff_case)
            while tot < 100:
                c1_fname, c2_fname = pair
                c1, c2 = (
                    pickle.load(open(c1_fname, "rb")),
                    pickle.load(open(c2_fname, "rb"))
                )
                tot += 1
                run_corrs.append(ridge_fit(c1, c2))
            diff_cases[int(c)] = sum(run_corrs) / tot

        pickle.dump(diff_cases, open(f"{out_name}_diff.pkl", "wb"))

        sys.exit(0)



    cases = get_cases()
    case = [c for c in cases if c.__class__.__name__[4:] in str(case_id)][0]

    tf = HookedTransformer(pickle.load(open(CFG_PATH, "rb")))

    model_path = ROOT / f"{case_id}-{model_id}" / "ll_model_510.pth"

    tf.load_state_dict(torch.load(model_path))

    

    # get the clean data (exactly 200 samples)
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


    # intervene on all of the attention heads individually and then get the accumulated residuals

    cdata = case.get_corrupted_data(min_samples=200, max_samples=200)
    corr = case.get_correspondence()
    cloader = cdata.make_loader(batch_size=batch_size)
    intervene(corr_data, tf)




if __name__ == "__main__":
    fire.Fire(main)