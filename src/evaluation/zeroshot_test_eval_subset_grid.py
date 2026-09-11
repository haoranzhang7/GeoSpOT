"""Evaluate every subset-selection config against one shared test dataloader."""

import argparse
import logging
import os
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
    p.add_argument('--test_subset_size', type=int, help='Overrides the per-config val_subset_size default')
    p.add_argument('--seeds', type=int, nargs="+", default=[6651033, 9272605, 1206448, 2180968, 114325])
    p.add_argument('--tgt_domain', type=str, default='all')
    p.add_argument('--k_values', type=int, nargs="+", default=[1, 2, 5])
    p.add_argument('--budget_values', type=int, nargs="+", default=[2000, 5000, 10000])
    p.add_argument('--ot_embedding_types', type=str, nargs="+", default=['bert', 'geoclip', 'satclip', 'geodesic'])
    p.add_argument('--combined_embeddings', type=str, nargs="+", default=['geoclip', 'satclip', 'geodesic'])
    p.add_argument('--combined_lambda', type=str, default='0.5')
    args = p.parse_args()

    log_dir = os.path.join(args.log_root, "2_zeroshot_eval_subset", args.model)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"zeroshot_eval_subset_grid_{args.domain_type}_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(log_path)
    logger = logging.getLogger(__name__)
    print(f"Log stored at {log_path}")

    print("Loading Dataset...")
    dataset = load_dataset(args.dataset, root_dir=args.data_dir)

    # One test dataloader per distinct val_subset_size (each config defaults its
    # test_subset_size to its own val_subset_size), unless overridden globally.
    sizes = {args.test_subset_size} if args.test_subset_size is not None else {b // 2 for b in args.budget_values}
    print(f"Loading Test Dataloader(s) for target domain {args.tgt_domain}, sizes {sizes}...")
    shared_dataloaders = {}
    for size in sizes:
        g = setup_seeds(args.seeds[0])
        shared_dataloaders[(args.tgt_domain, size)] = setup_test_dataloader(
            args.dataset, dataset, args.tgt_domain, args.eval_batch_size, g, args.seeds[0], test_subset_size=size)

    configs = build_configs(args)
    for i, subset_params in enumerate(configs):
        print(f"[{i + 1}/{len(configs)}] {subset_params}")
        main(args, [args.tgt_domain], subset_params,
             dataset=dataset, shared_dataloaders=shared_dataloaders, logger=logger)
