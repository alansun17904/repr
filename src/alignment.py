import os
import re
import sys
import tqdm
import fire
import torch
import pickle
import random
import itertools
import numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
from transformer_lens import HookedTransformer
from circuits_benchmark.utils.get_cases import get_cases


CFG_PATH = "ll_model_cfg_510.pkl"
EVAL_PATH = "eval-accuracy.pkl"


def ridge_fit(X, Y):
    rcv = RidgeCV(alpha_per_target=True, fit_intercept=False)
    rcv.fit(X, Y)
    return rcv.score(X, Y)


def main(dir_name, out_name, n_comps=10, batch_size=256, intervene=False, baseline_acc=0.8):

    # check to see if the evaluation map exists
    dir_name = Path(dir_name)
    if not os.path.exists(dir_name / EVAL_PATH):
        raise ValueError(f"Evaluation map `{str(dir_name)} / eval-accuracy.pkl` does not exist")

    evalu = pickle.load(open(dir_name / EVAL_PATH, "rb"))
    # get all of the case-seed pairs that are above baseline_acc
    
    case_seeds = list(filter(lambda x: x[1][0] >= baseline_acc, evalu.items()))
    case_seeds = [v[0] for v in case_seeds] 

    case2seed = dict()
    for cs in case_seeds:
        if cs[0] not in case2seed:
            case2seed[cs[0]] = [cs[1]]
        else:
            case2seed[cs[0]].append(cs[1])

    if not intervene:

        case2alignment = dict()
        _cases = list(case2seed.keys())
        _len = len(case2seed)

        pbar = tqdm.tqdm(total=_len * (_len - 1))

        for k1 in range(len(case2seed.items())):
            case1 = _cases[k1]
            c1 = random.choices(case2seed[case1], k=n_comps)
            c1_reprs = [
                pickle.load(open(dir_name / f"c{case1}-s{s1}-reprs.pkl", "rb")).cpu().detach().numpy()
                for s1 in c1
            ]

            for k2 in range(k1, len(case2seed.items())):
                case2 = _cases[k2]
                c2 = random.choices(case2seed[case2], k=n_comps)
                c2_reprs = [
                    pickle.load(open(dir_name / f"c{case2}-s{s2}-reprs.pkl", "rb")).cpu().detach().numpy()
                    for s2 in c2
                ]

                case2alignment[(_cases[k1], _cases[k2])] = [ridge_fit(c1_reprs[i], c2_reprs[i]) for i in range(n_comps)]
                pbar.update(1)

            pickle.dump(case2alignment, open(f"{out_name}", "wb"))
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
