"""
Domain adaptation pretraining script.
Supports both vision models (ResNet, DenseNet) on iNaturalist/FMOW 
and text models (BERT) on GeoYFCCText.
"""

import argparse
from base_trainer import BasePretrainTrainer, load_config


class PretrainTrainer(BasePretrainTrainer):
    """Standard pretraining trainer without subset selection."""
    pass


def main(config, pretrain_domain_idx, model_seed):
    """Main training function."""
    trainer = PretrainTrainer(config, pretrain_domain_idx, model_seed)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--pretrain_domain', type=int, required=True, help='Domain to pretrain model')
    parser.add_argument('-s', '--model_seed', type=int, required=True, help='Model seed')
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    CONFIG = load_config(args.config)
    main(CONFIG, args.pretrain_domain, args.model_seed)