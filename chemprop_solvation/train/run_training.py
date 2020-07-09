from argparse import Namespace
import csv
from logging import Logger
import os
from pprint import pformat
from typing import List

import numpy as np
from tensorboardX import SummaryWriter
import torch
from tqdm import trange
import pickle
from torch.optim.lr_scheduler import ExponentialLR

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from chemprop_solvation.data import StandardScaler
from chemprop_solvation.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop_solvation.models import build_model
from chemprop_solvation.nn_utils import param_count
from chemprop_solvation.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint
from rdkit import Chem
import matplotlib.pyplot as plt


def run_training(args: Namespace, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # Print args
    debug(pformat(vars(args)))

    # Get data
    debug('Loading data')
    args.task_names = get_task_names(path=args.data_path, solvation=args.solvation)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')

    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args, features_path=args.separate_test_features_path, logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args, features_path=args.separate_val_features_path, logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type, sizes=(0.9, 0.1, 0.0), seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type, sizes=(0.9, 0.1, 0.0), seed=args.seed, args=args, logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.save_smiles_splits or args.save_inchi_split:
        with open(args.data_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            lines_by_smiles = {}
            lines_by_solvation_combo = {}
            indices_by_smiles = {}
            indices_by_solvation_combo = {}
            for i, line in enumerate(reader):
                smiles = line[0]
                if "InChI" in smiles:
                    inchi = smiles
                    mol = Chem.MolFromInchi(inchi)
                    smiles = Chem.MolToSmiles(mol)
                else:
                    mol = Chem.MolFromSmiles(smiles)
                    inchi = Chem.MolToInchi(mol)
                if args.solvation:
                    solvent_smiles = smiles
                    solvent_inchi = inchi
                    solute_smiles = line[1]
                    if "InChI" in solute_smiles:
                        solute_inchi = solute_smiles
                        solute_mol = Chem.MolFromInchi(solute_inchi)
                        solute_smiles = Chem.MolToSmiles(solute_mol)
                    else:
                        solute_mol = Chem.MolFromSmiles(solute_smiles)
                        solute_inchi = Chem.MolToInchi(solute_mol)
                    solvation_set_smiles = [solvent_smiles, solute_smiles]

                lines_by_smiles[smiles] = line
                indices_by_smiles[smiles] = i
                if args.solvation:
                    lines_by_solvation_combo[solvation_set_smiles] = line
                    indices_by_solvation_combo[solvation_set_smiles] = i

        split_string = ""
        if args.save_smiles_split:
            split_string = "smiles"
        else:
            split_string = "inchi"

        all_split_indices = []

        for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
            with open(os.path.join(args.save_dir, name + '_' + split_string + '.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow([split_string])
                if split_string == "smiles":
                    if args.solvation:
                        for smiles_set in dataset.solvation_set_smiles():
                            writer.writerow(smiles_set)
                    else:
                        for smiles in dataset.smiles():
                            writer.writerow([smiles])
                else:
                    if args.solvation:
                        for inchi_set in dataset.solvation_set_inchi():
                            writer.writerow([inchi_set])
                    else:
                        for inchi in dataset.inchi():
                            writer.writerow([inchi])

            with open(os.path.join(args.save_dir, name + '_full.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                if args.solvation:
                    for smiles_set in dataset.solvation_set_smiles():
                        writer.writerow(lines_by_smiles[smiles_set])
                else:
                    for smiles in dataset.smiles():
                        writer.writerow(lines_by_smiles[smiles])

            split_indices = []
            if args.solvation:
                for smiles_set in dataset.solvation_set_smiles():
                    split_indices.append(indices_by_solvation_combo[smiles_set])
                    split_indices = sorted(split_indices)
            else:
                for smiles in dataset.smiles():
                    split_indices.append(indices_by_smiles[smiles])
                    split_indices = sorted(split_indices)

            all_split_indices.append(split_indices)
        with open(os.path.join(args.save_dir, 'split_indices.pckl'), 'wb') as f:
            pickle.dump(all_split_indices, f)

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)
    
    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        if args.solvation:
            train_smiles, train_targets = train_data.solvation_set_smiles(), train_data.targets()
        else:
            train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
    else:
        scaler = None

    # Get loss and metric functions
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)
    metric_func_rmse = get_metric_func(metric='rmse')
    metric_func_mae = get_metric_func(metric='mae')
    metric_func_r2 = get_metric_func(metric='r2')

    # Set up test set evaluation
    if args.solvation:
        test_smiles, test_targets = test_data.solvation_set_smiles(), test_data.targets()
    else:
        test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == 'multiclass':
        # not adjusted for solvation
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        # save in other file for plots
        file_summary_FV = open(os.path.join(args.save_dir, "epochs_loss.txt"),'w+')
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)
        # Load/build model
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(args)

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning cudrate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in trange(args.epochs):
            debug(f'Epoch {epoch}')

            (n_iter, loss_sum) = train(
                model=model,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer
            )

            if isinstance(scheduler, ExponentialLR):
                scheduler.step()

            val_scores = evaluate(
                model=model,
                data=val_data,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )

            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            debug(f'Validation {args.metric} = {avg_val_score:.6f}')
            writer.add_scalar(f'validation_{args.metric}', avg_val_score, n_iter)
            writer.add_scalar(f'loss_epoch', loss_sum, epoch)
            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
                    writer.add_scalar(f'validation_{task_name}_{args.metric}', val_score, n_iter)
            file_summary_FV.write(str(epoch)+" "+str(loss_sum)+" "+str(avg_val_score)+"\n")
            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)        

        # Evaluate on test set using model with best validation score
        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)

        train_preds = predict(
            model=model,
            data=train_data,
            batch_size=args.batch_size,
            scaler=scaler
        )

        val_preds = predict(
            model=model,
            data=val_data,
            batch_size=args.batch_size,
            scaler=scaler
        )

        test_preds = predict(
            model=model,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler
        )

        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )

        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds)

        # Average test score
        avg_test_score = np.nanmean(test_scores)
        info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')
        writer.add_scalar(f'test_{args.metric}', avg_test_score, 0)

        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, test_scores):
                info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
                writer.add_scalar(f'test_{task_name}_{args.metric}', test_score, n_iter)

        if args.detailed_results:
            f2 = open(os.path.join(args.save_dir, 'summary_results_model_'+str(model_idx)+'.csv'), 'w')
            writer2 = csv.writer(f2)
            writer2.writerow(["dataset", "rmse", "mae", "max"])
            for dataset, name, pred in [(train_data, 'train', train_preds), (val_data, 'val', val_preds),
                                        (test_data, 'test', test_preds)]:
                with open(os.path.join(args.save_dir, name + '_results_model_'+str(model_idx)+'.csv'), 'w') as f:
                    writer_summary = csv.writer(f)
                    rmse = 0
                    mae = 0
                    maxe = 0
                    if args.solvation:
                        writer_summary.writerow(["smiles_solvent", "smiles_solute",
                                                 "targets", "predictions", "diff", "diff^2"])
                        for count in range(len(dataset)):
                            mae_part = np.abs(dataset.targets()[count][0]-pred[count][0])
                            rmse_part = np.power(dataset.targets()[count][0]-pred[count][0], 2)
                            writer_summary.writerow([dataset.solvation_set_smiles()[count][0],
                                                     dataset.solvation_set_smiles()[count][1],
                                                     dataset.targets()[count][0], pred[count][0],
                                                     mae_part, rmse_part])
                            rmse += rmse_part
                            mae += mae_part
                            maxe = mae_part if mae_part > maxe else maxe
                            # need to be adjusted for the case that you have more targets
                    else:
                        writer_summary.writerow(["smiles", "targets", "predictions"])
                        for count in range(len(dataset)):
                            mae_part = np.abs(dataset.targets()[count][0]-pred[count][0])
                            rmse_part = np.power(dataset.targets()[count][0]-pred[count][0], 2)
                            writer_summary.writerow([dataset.smiles()[count],
                                                     dataset.targets()[count][0], pred[count][0],
                                                     mae_part, rmse_part])
                            rmse += rmse_part
                            mae += mae_part
                            maxe = mae_part if mae_part > maxe else maxe
                    rmse = np.sqrt(rmse/len(dataset))
                    mae = mae/len(dataset)
                    writer2.writerow([name, round(rmse, 2), round(mae, 2), round(maxe, 2)])
                    fig, ax = plt.subplots()
                    ax.plot(dataset.targets()[:], pred[:], 'b.')
                    ax.set(xlabel='targets', ylabel='predictions')
                    fig.savefig(os.path.join(args.save_dir, name + '_parity_model_'+str(model_idx)+'.png'))


    # Evaluate ensemble on test set
    avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

    ensemble_scores = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=args.num_tasks,
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        logger=logger

    )

    # Average ensemble score
    avg_ensemble_test_score = np.nanmean(ensemble_scores)
    info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')
    writer.add_scalar(f'ensemble_test_{args.metric}', avg_ensemble_test_score, 0)

    # Individual ensemble scores
    if args.show_individual_scores:
        for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
            info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')

    return ensemble_scores
