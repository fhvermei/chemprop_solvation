import os
import numpy as np
from copy import deepcopy
from typing import Callable, Tuple, List

from chemprop_solvation.train.predict import predict
from chemprop_solvation.utils import load_args, load_checkpoint, load_scalers
from chemprop_solvation.data import MoleculeDatapoint, MoleculeDataset


def load_ML_estimator(model_dir: str) -> Callable[[List], Tuple[List, List, List]]:
    """
    Load the given ML model and return a function for evaluating it.

    :param model_dir: path to the ML model, either DirectML_Gsolv, DirectML_Hsolv, or SoluteML.
    :return estimator: a function "estimator", which takes a list of solute or solvent-solute SMILES as input
             and returns the predicted solvation properties.
    """
    # find the number of folds in the model directory and get the path to each fold
    fold_count = len([file for file in os.listdir(model_dir) if file.startswith('fold_')])
    fold_path = [os.path.join(model_dir, 'fold_' + str(f)) for f in range(fold_count)]
    # find the number of models in each fold
    model_count = len([file for file in os.listdir(fold_path[0]) if file.startswith('model_')])

    # Load all models and scaler for each fold
    fold_scaler_model_dict = {}
    for fold in fold_path:
        # Get the path to each model in the fold.
        model_path = [os.path.join(fold, 'model_' + str(m), 'model.pt') for m in range(model_count)]

        # Load the arguments and scalers to normalize predictions and features.
        # Each fold has the same arguments and scalers, so only load from the first model.
        train_args = load_args(model_path[0])
        scaler, features_scaler = load_scalers(model_path[0])

        # Load models in ensemble
        models = []
        for checkpoint_path in model_path:
            models.append(load_checkpoint(checkpoint_path, cuda=False))

        fold_scaler_model_dict[fold] = {'train_args': train_args,
                                        'scaler': scaler,
                                        'features_scaler': features_scaler,
                                        'models': models}

    # Set up estimator
    def estimator(smiles: list(list()) = None):
        """
        Type 1. train_args.solvation == True:
            Computes the desired solvation property using the DirectML model.
            Supported properties are 'Gsolv' and 'Hsolv' in kcal/mol.
            Input: a list of a list or tuple containing solvent and solute smiles
            e.g. smiles = [[solvent1, solute1], [solvent2, solute2]], .. ]

        Type 2. train_args.solvation == False:
            Computes the solute parameters, E, S, A, B, L for the given solute SMILES using the SoluteML model.
            Input: a list of a list or tuple containing solute smiles
            e.g. smiles = [[solute1], [solute2]], .. ]

        :param smiles: a list of a list or tuple containing either solvent and solute smiles or just solute smiles.
        :return avg_pre: average predictions for the given SMILES.
        :return epi_unc: variance on prediction (=epistemic or model uncertainty)
        :return valid_indices: a list of valid SMILES indices that could be used for the prediction.
        """
        # Load the train argument. It's same for all folds, so use the first model of the first fold.
        train_args = load_args(os.path.join(fold_path[0], 'model_0', 'model.pt'))

        # Set the feature generator to 'rdkit_2d_normalized' if train_args.solvation is False
        if not train_args.solvation:
            train_args.features_generator = ['rdkit_2d_normalized']

        # Convert the smiles, remove invalid smiles from the list
        data = MoleculeDataset([MoleculeDatapoint(sm, train_args) for sm in smiles])

        # Valid indices is a list of smiles index that could be calculated by the ML models
        if train_args.solvation:
            valid_indices = [i for i in range(len(data)) if
                             data[i].solvent_mol is not None and data[i].solute_mol is not None]
        else:
            valid_indices = [i for i in range(len(data)) if data[i].mol is not None]
        data = MoleculeDataset([data[i] for i in valid_indices])

        # Check that there is at least one data.
        if len(data) == 0:
            raise ValueError(f'No valid smiles are given.')

        # Get the prediction from each model in each fold.
        all_preds = []
        for fold in fold_path:
            # Get the arguments, scalers, and models of the fold
            train_args = fold_scaler_model_dict[fold]['train_args']
            scaler = fold_scaler_model_dict[fold]['scaler']
            features_scaler = fold_scaler_model_dict[fold]['features_scaler']
            models = fold_scaler_model_dict[fold]['models']

            # Normalize features. Use deepcopy so the original data are not changed when used for different folds.
            data_per_fold = deepcopy(data)
            if train_args.features_scaling:
                data_per_fold.normalize_features(features_scaler)

            # make predictions for all models
            for model in models:
                model_preds = predict(
                    model=model,
                    data=data_per_fold,
                    batch_size=1,
                    scaler=scaler
                )
                all_preds.append(np.array(model_preds))

        # calculate average prediction and variance on prediction (=epistemic or model uncertainty)
        epi_unc = []
        avg_pre = []
        all_preds = np.array(all_preds)
        for i in range(len(valid_indices)):
            if train_args.solvation:
                epi_unc.append(np.var(all_preds[:, i]))
                avg_pre.append(np.mean(all_preds[:, i]))
            else:
                epi_unc_per_data = []
                ave_pre_per_data = []
                for j in range(5):  # the 5 predictions correspond to E, S, A, B, and L.
                    epi_unc_per_data.append(np.var(all_preds[:, i, j]))
                    ave_pre_per_data.append(np.mean(all_preds[:, i, j]))
                epi_unc.append(epi_unc_per_data)
                avg_pre.append(ave_pre_per_data)

        return avg_pre, epi_unc, valid_indices

    return estimator

# Get the absolute path to the ML model directory
current_dir = os.path.abspath(os.path.dirname(__file__))
path_to_ML_models = os.path.join(current_dir, '../final_models')

def load_DirectML_Gsolv_estimator():
    """
    Load the DirectML_Gsolv model and return a evaluator function for it.
    The returned evaluator computes the solvation free energy using the DirectML_Gsolv model.
    For the evaluator:
        Input: a list of a list or tuple containing solvent and solute smiles
               e.g. smiles = [[solvent1, solute1], [solvent2, solute2]], .. ]
        Output 1: average solvation free energy prediction for each solvent-solute pair in kcal/mol.
        Output 2: variance on predictions (=epistemic or model uncertainty) for each solvent-solute pair in kcal/mol.
        Output 3: a list of valid SMILES indices that could be used for the prediction.
    """
    path_to_model = os.path.join(path_to_ML_models, 'DirectML_Gsolv')
    return load_ML_estimator(path_to_model)

def load_DirectML_Hsolv_estimator():
    """
    Load the DirectML_Hsolv model and return a evaluator function for it.
    The returned evaluator computes the solvation enthalpy using the DirectML_Hsolv model.
    For the evaluator:
        Input: a list of a list or tuple containing solvent and solute smiles
               e.g. smiles = [[solvent1, solute1], [solvent2, solute2]], .. ]
        Output 1: average solvation enthalpy prediction fpr each solvent-solute pair in kcal/mol.
        Output 2: variance on predictions (=epistemic or model uncertainty) for each solvent-solute pair in kcal/mol.
        Output 3: a list of valid SMILES indices that could be used for the prediction.
    """
    path_to_model = os.path.join(path_to_ML_models, 'DirectML_Hsolv')
    return load_ML_estimator(path_to_model)

def load_SoluteML_estimator():
    """
    Load the SoluteML model and return a evaluator function for it.
    The returned evaluator computes the solute parameters, E, S, A, B, L using the SoluteML model.
    For the evaluator:
        Input: a list of a list or tuple containing solute smiles
             e.g. smiles = [[solute1], [solute2]], .. ]
        Output 1: average solute parameter predictions (E, S, A, B, L) for each solute species.
        Output 2: variance on predictions (=epistemic or model uncertainty) for each solute species.
        Output 3: a list of valid SMILES indices that could be used for the prediction.
    """
    path_to_model = os.path.join(path_to_ML_models, 'SoluteML')
    return load_ML_estimator(path_to_model)
