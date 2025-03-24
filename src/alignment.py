import os
import re
import sys
import tqdm
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


def main(case_id, model_id, out_name, batch_size=256, intervene=False):

    if not intervene:
        print("Not using arguments: case_id, model_id, batch_size")
        ROOT = Path("src/data/reprs/")
        files = os.listdir(ROOT)
        
        cases = set()

        for f_name in files:
            case_id = re.search(r"var_case_(\d+)", f_name)
            if case_id is not None:
                cases.add(case_id.group(1))

        cases = list(cases)

        print(cases)

        # compare all of the implementaitons within the same class
        same_cls_corr = dict()
        for c in tqdm.tqdm(cases):
            same_case = list(filter(lambda x: x.startswith(f"var_case_{c}"), os.listdir(ROOT)))

            dps = [
                pickle.load(open(ROOT / v, "rb")).cpu().detach().numpy()
                for v in same_case
            ]

            print(f"Found {len(dps)} of the same case.")

            run_corrs = []
            tot = 0

            same_generator = itertools.combinations(dps, 2)
            while tot < 10:
                try:
                    pair = next(same_generator)
                except StopIteration:
                    break
                tot += 1

                if np.isnan(pair[0]).any() or np.isnan(pair[1]).any():
                    print("There exists NaNs in [0] or [1] of pair, so skipping!")
                    continue

                run_corrs.append(ridge_fit(pair[0], pair[1]))

                if len(run_corrs) != 0:
                    same_cls_corr[int(c)] = (sum(run_corrs) / tot, np.std(run_corrs))

                pickle.dump(same_cls_corr, open(f"{out_name}_same.pkl", "wb"))

        # compare alignment across different tasks
        diff_cases = dict()
        for c in cases:
            same_case = list(filter(lambda x: x.startswith(f"var_case_{c}"), os.listdir(ROOT)))
            diff_case = list(filter(lambda x: not x.startswith(f"var_case_{c}"), os.listdir(ROOT)))
            tot = 0
            run_corrs = []

            scs = [
                pickle.load(open(ROOT / v, "rb")).cpu().detach().numpy()
                for v in same_case
            ]

            dcs = [
                pickle.load(open(ROOT / v, "rb")).cpu().detach().numpy()
                for v in diff_case
            ]

            cross_generator = itertools.product(scs, dcs)
            with tqdm.tqdm(total=10) as pbar:
                while tot < 10:
                    try:
                        pair = next(cross_generator)
                    except StopIteration:
                        break
                    c1, c2 = pair

                    if np.isnan(c1).any() or np.isnan(c2).any():
                        print("There exists NaNs in [0] or [1] of pair, so skipping!")
                        continue

                    tot += 1
                    run_corrs.append(ridge_fit(c1, c2))
                    pbar.update(1)
                diff_cases[int(c)] = (sum(run_corrs) / tot, np.std(run_corrs))
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
