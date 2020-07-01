from argparse import Namespace
import random
from typing import Callable, List, Union

import numpy as np
from torch.utils.data.dataset import Dataset
from rdkit import Chem

from .scaler import StandardScaler
from chemprop_solvation.features import get_features_generator


class MoleculeDatapoint:
    """A MoleculeDatapoint contains a single molecule and its associated features and targets."""
    """In case of solvation, it contains 2 molecules, solvent and solute"""

    def __init__(self,
                 line: List[str],
                 args: Namespace = None,
                 features: np.ndarray = None,
                 use_compound_names: bool = False):
        """
        Initializes a MoleculeDatapoint, which contains a single molecule.

        :param line: A list of strings generated by separating a line in a data CSV file by comma.
        :param args: Arguments.
        :param features: A numpy array containing additional features (ex. Morgan fingerprint).
        :param use_compound_names: Whether the data CSV includes the compound name on each line.
        """
        if args is not None:
            self.features_generator = args.features_generator
            self.args = args
            self.solvation = args.solvation
        else:
            self.features_generator = self.args = None

        if features is not None and self.features_generator is not None:
            raise ValueError('Currently cannot provide both loaded features and a features generator.')

        self.features = features

        # compound_names not enabled for solvation
        if use_compound_names:
            self.compound_name = line[0]  # str
            line = line[1:]
        else:
            self.compound_name = None

        # assume only one solvent for now, no solvent mixtures
        if self.solvation:
            if "InChI" in line[0]:
                self.solvent_inchi = line[0]
                self.solvent_mol = Chem.MolFromInchi(self.solvent_inchi)
                self.solvent_smiles = Chem.MolToSmiles(self.solvent_mol)  # str
            else:
                self.solvent_smiles = line[0]
                self.solvent_mol = Chem.MolFromSmiles(self.solvent_smiles)
                self.solvent_inchi = Chem.MolToInchi(self.solvent_mol)

            if "InChI" in line[1]:
                self.solute_inchi = line[1]
                self.solute_mol = Chem.MolFromInchi(self.solute_inchi)
                self.solute_smiles = Chem.MolToSmiles(self.solute_mol)  # str
            else:
                self.solute_smiles = line[1]
                self.solute_mol = Chem.MolFromSmiles(self.solute_smiles)
                self.solute_inchi = Chem.MolToInchi(self.solute_mol)
            self.solvation_set_inchi = [self.solvent_inchi, self.solute_inchi]
            self.solvation_set_smiles = [self.solvent_smiles, self.solute_smiles]

        if "InChI" in line[0]:
            print(line[0])
            self.inchi = line[0]
            self.mol = Chem.MolFromInchi(self.inchi)
            self.smiles = Chem.MolToSmiles(self.mol)  # str
        else:
            self.smiles = line[0]
            self.mol = Chem.MolFromSmiles(self.smiles)
            self.inchi = Chem.MolToInchi(self.mol)
            # use standard inchi for now, this should be extended to inchi with fixed H layer

        # Generate additional features if given a generator
        # not yet enabled for solvation
        if self.features_generator is not None:
            self.features = []

            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                if self.mol is not None and self.mol.GetNumHeavyAtoms() > 0:
                    self.features.extend(features_generator(self.mol))

            self.features = np.array(self.features)

        # Fix nans in features
        if self.features is not None:
            replace_token = 0
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

        # Create targets
        if self.solvation:
            self.targets = [float(x) if x != '' else None for x in line[2:]]
        else:
            self.targets = [float(x) if x != '' else None for x in line[1:]]

    def set_features(self, features: np.ndarray):
        """
        Sets the features of the molecule.

        :param features: A 1-D numpy array of features for the molecule.
        """
        self.features = features

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[float]):
        """
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        """
        self.targets = targets



class MoleculeDataset(Dataset):
    """A MoleculeDataset contains a list of molecules and their associated features and targets."""

    def __init__(self, data: List[MoleculeDatapoint]):
        """
        Initializes a MoleculeDataset, which contains a list of MoleculeDatapoints (i.e. a list of molecules and their targets).

        :param data: A list of MoleculeDatapoints.
        """
        self.data = data
        self.args = self.data[0].args if len(self.data) > 0 else None
        self.scaler = None
        self.solvation = self.args.solvation

    def compound_names(self) -> List[str]:
        """
        Returns the compound names associated with the molecule (if they exist).

        :return: A list of compound names or None if the dataset does not contain compound names.
        """
        if len(self.data) == 0 or self.data[0].compound_name is None:
            return None

        return [d.compound_name for d in self.data]

    def smiles(self) -> List[str]:
        """
        Returns the smiles strings associated with the molecules.

        :return: A list of smiles strings.
        """
        return [d.smiles for d in self.data]

    def solvent_smiles(self) -> List[str]:
        """
        Returns the smiles strings associated with the molecules.

        :return: A list of smiles strings.
        """
        return [d.smiles_solvent for d in self.data]

    def solute_smiles(self) -> List[str]:
        """
        Returns the smiles strings associated with the molecules.

        :return: A list of smiles strings.
        """
        return [d.smiles_solute for d in self.data]

    def solvation_set_smiles(self) -> List[List[str]]:
        """
        Returns the solvent and solute smiles strings associated with the solvation pairs.

        :return: A list of a list of smiles strings.
        """
        return [d.solvation_set_smiles for d in self.data]

    def inchi(self) -> List[str]:
        """
        Returns the inchi strings associated with the molecules.

        :return: A list of inchi strings.
        """
        return [d.inchi for d in self.data]

    def solvent_inchi(self) -> List[str]:
        """
        Returns the inchi strings associated with the molecules.

        :return: A list of inchi strings.
        """
        return [d.inchi_solvent for d in self.data]

    def solute_inchi(self) -> List[str]:
        """
        Returns the inchi strings associated with the molecules.

        :return: A list of inchi strings.
        """
        return [d.inchi_solute for d in self.data]

    def solvation_set_inchi(self) -> List[List[str]]:
        """
        Returns the solvent and solute inchi strings associated with the solvation pairs.

        :return: A list of a list of inchi strings.
        """
        return [d.solvation_set_inchi for d in self.data]

    def mols(self) -> List[Chem.Mol]:
        """
        Returns the RDKit molecules associated with the molecules.

        :return: A list of RDKit Mols.
        """
        return [d.mol for d in self.data]

    def solvent_mols(self) -> List[Chem.Mol]:
        """
        Returns the RDKit molecules associated with the molecules.

        :return: A list of RDKit Mols.
        """
        return [d.mol_solvent for d in self.data]

    def solute_mols(self) -> List[Chem.Mol]:
        """
        Returns the RDKit molecules associated with the molecules.

        :return: A list of RDKit Mols.
        """
        return [d.mol_solute for d in self.data]

    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self.data) == 0 or self.data[0].features is None:
            return None

        return [d.features for d in self.data]

    def targets(self) -> List[List[float]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats containing the targets.
        """
        return [d.targets for d in self.data]

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return self.data[0].num_tasks() if len(self.data) > 0 else None

    def features_size(self) -> int:
        """
        Returns the size of the features array associated with each molecule.

        :return: The size of the features.
        """
        return len(self.data[0].features) if len(self.data) > 0 and self.data[0].features is not None else None

    def shuffle(self, seed: int = None):
        """
        Shuffles the dataset.

        :param seed: Optional random seed.
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)
    
    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> StandardScaler:
        """
        Normalizes the features of the dataset using a StandardScaler (subtract mean, divide by standard deviation).

        If a scaler is provided, uses that scaler to perform the normalization. Otherwise fits a scaler to the
        features in the dataset and then performs the normalization.

        :param scaler: A fitted StandardScaler. Used if provided. Otherwise a StandardScaler is fit on
        this dataset and is then used.
        :param replace_nan_token: What to replace nans with.
        :return: A fitted StandardScaler. If a scaler is provided, this is the same scaler. Otherwise, this is
        a scaler fit on this dataset.
        """
        if len(self.data) == 0 or self.data[0].features is None:
            return None

        if scaler is not None:
            self.scaler = scaler

        elif self.scaler is None:
            features = np.vstack([d.features for d in self.data])
            self.scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self.scaler.fit(features)

        for d in self.data:
            d.set_features(self.scaler.transform(d.features.reshape(1, -1))[0])

        return self.scaler
    
    def set_targets(self, targets: List[List[float]]):
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats containing targets for each molecule. This must be the
        same length as the underlying dataset.
        """
        assert len(self.data) == len(targets)
        for i in range(len(self.data)):
            self.data[i].set_targets(targets[i])

    def sort(self, key: Callable):
        """
        Sorts the dataset using the provided key.

        :param key: A function on a MoleculeDatapoint to determine the sorting order.
        """
        self.data.sort(key=key)

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e. the number of molecules).

        :return: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        """
        Gets one or more MoleculeDatapoints via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A MoleculeDatapoint if an int is provided or a list of MoleculeDatapoints if a slice is provided.
        """
        return self.data[item]

    def is_solvation(self) -> bool:
        """
        Returns if you are dealing with solvation
        :return: boolean
        """
        return self.solvation