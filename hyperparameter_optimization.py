"""Optimizes hyperparameters using Bayesian optimization."""

from argparse import ArgumentParser, Namespace
from copy import deepcopy
import json
from typing import Dict, Union
import os

from hyperopt import fmin, hp, tpe
import numpy as np

from chemprop_solvation.models import build_model
from chemprop_solvation.nn_utils import param_count
from chemprop_solvation.parsing import add_train_args, modify_train_args
from chemprop_solvation.train import cross_validate
from chemprop_solvation.utils import create_logger, makedirs


SPACE = {
    'hidden_size': hp.quniform('hidden_size', low=100, high=500, q=100),
    'depth': hp.quniform('depth', low=3, high=5, q=1),
    'dropout': hp.quniform('dropout', low=0.0, high=0.3, q=0.05),
    'ffn_num_layers': hp.quniform('ffn_num_layers', low=2, high=5, q=1),
    'ffn_hidden_size': hp.quniform('ffn_hidden_size', low=400, high=1000, q=200),
    'warmup_epochs': hp.quniform('warmup_epochs', low=2, high=6, q=2),
    'batch_size': hp.quniform('batch_size', low=10, high=100, q=10),
    'init_lr': hp.loguniform('init_lr', low=-11, high=-5),
    'max_lr': hp.loguniform('max_lr', low=-11, high=-5),
    'final_lr': hp.loguniform('final_lr', low=-14, high=-7)
}
INT_KEYS = ['hidden_size', 'depth', 'ffn_num_layers', 'ffn_hidden_size', 'warmup_epochs', 'batch_size']


def grid_search(args: Namespace):
    # Create loggers
    logger = create_logger(name='hyperparameter_optimization', save_dir=args.log_dir, quiet=True)
    train_logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)

    # Run grid search
    results = []

    # Define hyperparameter optimization
    def objective(hyperparams: Dict[str, Union[int, float]]) -> float:
        # Convert hyperparams from float to int when necessary
        for key in INT_KEYS:
            hyperparams[key] = int(hyperparams[key])

        # Update args with hyperparams
        hyper_args = deepcopy(args)
        if args.save_dir is not None:
            folder_name = '_'.join(f'{key}_{value}' for key, value in hyperparams.items())
            hyper_args.save_dir = os.path.join(hyper_args.save_dir, folder_name)
        for key, value in hyperparams.items():
            setattr(hyper_args, key, value)

        # Record hyperparameters
        logger.info(hyperparams)

        # Cross validate
        mean_score, std_score = cross_validate(hyper_args, train_logger)

        # Record results
        temp_model = build_model(hyper_args)
        num_params = param_count(temp_model)
        logger.info(f'num params: {num_params:,}')
        logger.info(f'{mean_score} +/- {std_score} {hyper_args.metric}')

        results.append({
            'mean_score': mean_score,
            'std_score': std_score,
            'hyperparams': hyperparams,
            'num_params': num_params
        })

        # Deal with nan
        if np.isnan(mean_score):
            if hyper_args.dataset_type == 'classification':
                mean_score = 0
            else:
                raise ValueError('Can\'t handle nan score for non-classification dataset.')

        return (1 if hyper_args.minimize_score else -1) * mean_score

    fmin(objective, SPACE, algo=tpe.suggest, max_evals=args.num_iters)

    # Report best result
    results = [result for result in results if not np.isnan(result['mean_score'])]
    best_result = min(results, key=lambda result: (1 if args.minimize_score else -1) * result['mean_score'])
    logger.info('best')
    logger.info(best_result['hyperparams'])
    logger.info(f'num params: {best_result["num_params"]:,}')
    logger.info(f'{best_result["mean_score"]} +/- {best_result["std_score"]} {args.metric}')

    # Save best hyperparameter settings as JSON config file
    makedirs(args.config_save_path, isfile=True)

    with open(args.config_save_path, 'w') as f:
        json.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    add_train_args(parser)
    parser.add_argument('--num_iters', type=int, default=50,
                        help='Number of hyperparameter choices to try')
    parser.add_argument('--config_save_path', type=str, required=True,
                        help='Path to .json file where best hyperparameter settings will be written')
    parser.add_argument('--log_dir', type=str,
                        help='(Optional) Path to a directory where all results of the hyperparameter optimization will be written')
    args = parser.parse_args()
    modify_train_args(args)

    grid_search(args)
