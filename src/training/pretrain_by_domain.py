"""
Domain adaptation pretraining script.
Supports both vision models (ResNet, DenseNet) on iNaturalist/FMOW 
and text models (BERT) on GeoYFCCText.
"""

import argparse
from base_trainer import BasePretrainTrainer


class PretrainTrainer(BasePretrainTrainer):
    """Standard pretraining trainer without subset selection."""
    pass


def main(args):
    """Main training function."""
    model_entry = {
        'LEARNING_RATE': args.lr, 'DEFAULT_LEARNING_RATE': args.lr,
        'NUM_EPOCHS': args.num_epochs,
        'OPTIMIZER': args.optimizer, 'DEFAULT_OPTIMIZER': args.optimizer,
        'WEIGHT_DECAY': args.weight_decay, 'DEFAULT_WEIGHT_DECAY': args.weight_decay,
        'SCHEDULER': args.scheduler,
    }
    config = {
        'DATASET_NAME': args.dataset,
        'DOMAIN_TYPE': args.domain_type,
        'MODEL_NAME': args.model,
        'DATA_DIR': args.data_dir,
        'CHECKPOINT_ROOT': args.checkpoint_root,
        'LOG_ROOT': args.log_root,
        'TRAIN_BATCH_SIZE': args.train_batch_size,
        'EVAL_BATCH_SIZE': args.eval_batch_size,
        'PATIENCE': args.patience,
        'START_FROM_EPOCH': args.start_from_epoch,
        'MODELS': {args.model: model_entry, args.model.upper(): model_entry},
    }
    trainer = PretrainTrainer(config, args.pretrain_domain, args.model_seed)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--pretrain_domain', type=int, required=True, help='Domain to pretrain model')
    parser.add_argument('-s', '--model_seed', type=int, required=True, help='Model seed')
    parser.add_argument('--dataset', default='geoyfcc_text')
    parser.add_argument('--domain_type', default='countries')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--checkpoint_root', default='./results/pretrain/checkpoints')
    parser.add_argument('--log_root', default='./results/pretrain/logs')
    parser.add_argument('--model', default='bert_singlelabel')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--scheduler', default='cosine')
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--start_from_epoch', type=int, default=0)
    main(parser.parse_args())