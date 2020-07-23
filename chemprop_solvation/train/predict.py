from typing import List

import torch
import torch.nn as nn
from tqdm import trange

from chemprop_solvation.data import MoleculeDataset, StandardScaler


def predict(model: nn.Module,
            data: MoleculeDataset,
            batch_size: int,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []
    ale_unc = []
    num_iters, iter_step = len(data), batch_size

    aleatoric = model.aleatoric

    for i in trange(0, num_iters, iter_step):
        # Prepare batch
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        if data.is_solvation():
            smiles_batch, features_batch = mol_batch.solvation_set_smiles(), mol_batch.solvation_set_features()
        else:
            smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        # Run model
        batch = smiles_batch

        if not aleatoric:
            with torch.no_grad():
                batch_preds = model(batch, features_batch)

            batch_preds = batch_preds.data.cpu().numpy()

            # Inverse scale if regression
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)

            # Collect vectors
            batch_preds = batch_preds.tolist()
            preds.extend(batch_preds)
        else:
            with torch.no_grad():
                batch_preds, batch_logvar = model(batch, features_batch)
                batch_var = torch.exp(batch_logvar)
            batch_preds = batch_preds.data.cpu().numpy()
            batch_ale_unc = batch_var.data.cpu().numpy()

            # Inverse scale if regression
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)
                batch_ale_unc = scaler.inverse_transform_variance(batch_ale_unc)

            # Collect vectors
            batch_preds = batch_preds.tolist()
            batch_ale_unc = batch_ale_unc.tolist()
            preds.extend(batch_preds)
            ale_unc.extend(batch_ale_unc)

    if not aleatoric:
        return preds, None
    else:
        return preds, ale_unc