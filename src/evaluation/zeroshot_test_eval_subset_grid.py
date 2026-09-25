"""Evaluate every subset-selection config against one shared test dataloader."""

import argparse
import logging
import os
import re
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..', '..')))

from src.data.load_datasets import load_dataset
from src.core.utils import setup_logging, setup_seeds
from src.evaluation.zeroshot_test_eval_subset import main, setup_test_dataloader

OT_DEFAULTS = {'ot_method': 'sinkhorn_log', 'ot_reg': '0.01', 'ot_iter': '1000',
               'ot_metric': 'cosine', 'ot_norm': 'max_per_domain'}


def build_configs(a):
    configs = []
    for k in a.k_values:
        for budget in a.budget_values:
            base = {'subset_size': budget, 'val_subset_size': budget // 2,
                    'num_domains': k, 'tgt_domain': a.tgt_domain}
            configs.append({**base, 'domain_selection_method': 'random'})
            for emb in a.ot_embedding_types:
                configs.append({**base, 'domain_selection_method': 'ot',
                                 'ot_embedding_type': emb, **OT_DEFAULTS})
            for emb in a.combined_embeddings:
                configs.append({**base, 'domain_selection_method': 'ot',
                                 'ot_embedding_type': f"{emb}+bert", 'ot_lambda': a.combined_lambda,
                                 **OT_DEFAULTS})
    return configs


CKPT_DIR_RE = re.compile(r'^1_pretrain_subset(\d+)_K(\d+)_(ot|random)$')
_OT_SUFFIX = "_{ot_method}_{ot_reg}_{ot_iter}_{ot_metric}_{ot_norm}".format(**OT_DEFAULTS)


def _log_confirms_completion(log_dir, cfg_stem, seed):
    """True if any timestamped log for this (config, seed) contains the training-completed marker.

    Log filenames are `{cfg_stem}_seed{seed}_{timestamp}.log`; cfg_stem is identical to the
    checkpoint filename stem (everything before `_seed{seed}_best.pth`), since both are built
    from the same training-run naming convention in pretrain_by_domain_subset.py.
    """
    if not os.path.isdir(log_dir):
        return False
    prefix = f"{cfg_stem}_seed{seed}_"
    for fname in os.listdir(log_dir):
        if fname.startswith(prefix) and fname.endswith('.log'):
            try:
                with open(os.path.join(log_dir, fname)) as f:
                    if "Training completed in" in f.read():
                        return True
            except OSError:
                continue
    return False


def discover_configs(checkpoint_root, log_root, model_name, domain_type, tgt_domain):
    """Find every subset-selection training config on disk with a completed checkpoint for
    tgt_domain, regardless of which K/budget/embedding grid produced it (that grid has changed
    over time across experiments/07_subset_selection.sh revisions, so it can't be assumed).

    Returns a list of (subset_params, completed_seeds) tuples, one per unique training config
    (i.e. not per-seed); completed_seeds only includes seeds whose training log confirms
    "Training completed in", so an in-progress run's best-so-far checkpoint is excluded.
    """
    tgt_str = str(tgt_domain)
    prefix = f"pretrain_{domain_type}_{model_name}_subset"
    tgt_suffix = f"_tgt{tgt_str}"

    found = {}  # key -> (subset_params, cfg_stem, set of seeds)
    if not os.path.isdir(checkpoint_root):
        return []

    for dname in sorted(os.listdir(checkpoint_root)):
        m = CKPT_DIR_RE.match(dname)
        if not m:
            continue
        subset_size, k, method = int(m.group(1)), int(m.group(2)), m.group(3)
        model_dir = os.path.join(checkpoint_root, dname, model_name)
        if not os.path.isdir(model_dir):
            continue
        for norm_type in os.listdir(model_dir):
            norm_dir = os.path.join(model_dir, norm_type)
            if not os.path.isdir(norm_dir):
                continue
            for fname in os.listdir(norm_dir):
                fm = re.match(r'^(.+)_seed(\d+)_best\.pth$', fname)
                if not fm:
                    continue
                cfg_stem, seed = fm.group(1), int(fm.group(2))
                if not cfg_stem.startswith(prefix) or not cfg_stem.endswith(tgt_suffix):
                    continue

                body = cfg_stem[len(prefix):-len(tgt_suffix)]  # e.g. "2000_K2_OT_geoclip+bert_..._V1000"
                try:
                    b_str, rest = body.split('_K', 1)
                    k_str, rest = rest.split('_', 1)
                    method_part, val_str = rest.rsplit('_V', 1)
                    if int(b_str) != subset_size or int(k_str) != k or not val_str.isdigit():
                        continue
                except ValueError:
                    continue
                val_subset_size = int(val_str)

                base = {'subset_size': subset_size, 'val_subset_size': val_subset_size,
                        'num_domains': k, 'tgt_domain': tgt_domain}
                if method == 'random' and method_part == 'random':
                    subset_params = {**base, 'domain_selection_method': 'random'}
                elif method == 'ot' and method_part.startswith('OT_') and _OT_SUFFIX in method_part:
                    remainder = method_part[len('OT_'):]
                    emb, leftover = remainder.split(_OT_SUFFIX, 1)
                    ot_lambda = None
                    if leftover.startswith('_lambda'):
                        ot_lambda = leftover[len('_lambda'):]
                    elif leftover != '':
                        continue  # unrecognized trailing suffix
                    subset_params = {**base, 'domain_selection_method': 'ot',
                                      'ot_embedding_type': emb, 'ot_lambda': ot_lambda, **OT_DEFAULTS}
                else:
                    continue  # e.g. legacy naming (older embedding-type conventions), skip

                key = tuple(sorted(subset_params.items(), key=lambda kv: kv[0]))
                if key not in found:
                    found[key] = (subset_params, cfg_stem, set())
                found[key][2].add(seed)

    results = []
    for subset_params, cfg_stem, seeds in found.values():
        method = subset_params['domain_selection_method']
        log_dir = os.path.join(log_root, f"1_pretrain_subset{subset_params['subset_size']}_K{subset_params['num_domains']}_{method}", model_name)
        completed = sorted(s for s in seeds if _log_confirms_completion(log_dir, cfg_stem, s))
        if completed:
            results.append((subset_params, completed))
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='geoyfcc_text')
    p.add_argument('--domain_type', default='countries')
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--checkpoint_root', default='./results/subset/checkpoints')
    p.add_argument('--log_root', default='./results/subset/logs')
    p.add_argument('--results_root', default='./results/subset/test_results')
    p.add_argument('--model', default='bert_singlelabel')
    p.add_argument('--eval_batch_size', type=int, default=2048)
    p.add_argument('--summary_csv', type=str,
                    help="Path to the organized summary CSV (one row per config+seed eval, "
                         "appended across runs/targets). Defaults to "
                         "<results_root>/2_zeroshot_eval_subset/<model>/summary/"
                         "zeroshot_eval_<domain_type>_<model>_summary.csv. An eval already "
                         "present there (same target domain, config, seed, and test_subset_size) "
                         "is skipped.")
    p.add_argument('--test_subset_size', type=int,
                    help='Subsample the test set to this many examples instead of using the entire test split.')
    p.add_argument('--seeds', type=int, nargs="+", default=[6651033, 9272605, 1206448, 2180968, 114325])
    p.add_argument('--tgt_domain', type=str, default='all')
    p.add_argument('--k_values', type=int, nargs="+", default=[1, 2, 5])
    p.add_argument('--budget_values', type=int, nargs="+", default=[2000, 5000, 10000])
    p.add_argument('--ot_embedding_types', type=str, nargs="+", default=['bert', 'geoclip', 'satclip', 'geodesic'])
    p.add_argument('--combined_embeddings', type=str, nargs="+", default=['geoclip', 'satclip', 'geodesic'])
    p.add_argument('--combined_lambda', type=str, default='0.5')
    p.add_argument('--discover_checkpoints', action='store_true',
                    help="Instead of building the config grid from --k_values/--budget_values/--*_embedding_types, "
                         "scan --checkpoint_root/--log_root for every completed checkpoint for --tgt_domain "
                         "(only seeds whose training log has 'Training completed in'), across whatever grid "
                         "actually exists on disk. Use this for targets like a specific held-out domain where "
                         "the training grid has evolved across experiments/07_subset_selection.sh revisions.")
    args = p.parse_args()

    # Domain indices are compared against an int column (country_id); "all" is special-cased
    # to mean the pooled distribution, everything else must be an int or the domain mask ends
    # up comparing str to int and silently matches nothing.
    if args.tgt_domain != "all":
        args.tgt_domain = int(args.tgt_domain)

    log_dir = os.path.join(args.log_root, "2_zeroshot_eval_subset", args.model)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"zeroshot_eval_subset_grid_{args.domain_type}_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(log_path)
    logger = logging.getLogger(__name__)
    print(f"Log stored at {log_path}")

    print("Loading Dataset...")
    dataset = load_dataset(args.dataset, root_dir=args.data_dir)

    # One shared test dataloader for the target domain. Default (test_subset_size=None) is
    # the entire test split; pass --test_subset_size to subsample it instead.
    print(f"Loading Test Dataloader for target domain {args.tgt_domain}, "
          f"{'full test split' if args.test_subset_size is None else f'subsampled to {args.test_subset_size}'}...")
    g = setup_seeds(args.seeds[0])
    shared_dataloaders = {
        (args.tgt_domain, args.test_subset_size): setup_test_dataloader(
            args.dataset, dataset, args.tgt_domain, args.eval_batch_size, g, args.seeds[0],
            test_subset_size=args.test_subset_size)
    }

    if args.discover_checkpoints:
        discovered = discover_configs(args.checkpoint_root, args.log_root, args.model, args.domain_type, args.tgt_domain)
        print(f"Discovered {len(discovered)} completed training configs for target domain {args.tgt_domain}")
        for i, (subset_params, completed_seeds) in enumerate(discovered):
            print(f"[{i + 1}/{len(discovered)}] {subset_params} seeds={completed_seeds}")
            config_args = argparse.Namespace(**{**vars(args), 'seeds': completed_seeds})
            main(config_args, [args.tgt_domain], subset_params,
                 summary_csv_override=args.summary_csv,
                 dataset=dataset, shared_dataloaders=shared_dataloaders, logger=logger)
    else:
        configs = build_configs(args)
        for i, subset_params in enumerate(configs):
            print(f"[{i + 1}/{len(configs)}] {subset_params}")
            main(args, [args.tgt_domain], subset_params,
                 summary_csv_override=args.summary_csv,
                 dataset=dataset, shared_dataloaders=shared_dataloaders, logger=logger)
