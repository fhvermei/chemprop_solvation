from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import numpy as np

from chemprop_solvation.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop_solvation.features.featurization import mol2graph_solvation
from chemprop_solvation.nn_utils import index_select_ND, get_activation_function
from rdkit import Chem
#from rmgpy.molecule import Molecule
#import quantities
#import cython

class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args

        if self.features_only:
            return

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """

        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

            if self.features_only:
                return features_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        if self.atom_messages:
            a2a = mol_graph.get_a2a()
        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

            if self.atom_messages:
                a2a = a2a.cuda()

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size

        message = self.act_func(input)  # num_bonds x hidden_size
        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        if self.args.Tmelt:
            sssr = list()
            flatness = list()
            for mol in mol_graph.getsmiles():
                sssr.append(Chem.GetSSSR(Chem.MolFromSmiles(mol)))
                count_flatness = 0
                molecule = Chem.MolFromSmiles(mol)
                for i, atom in enumerate(molecule.GetAtoms()):
                    if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP or atom.GetIsAromatic():
                        count_flatness +=1
                count_flatness = count_flatness/(Chem.MolFromSmiles(mol).GetNumAtoms())
                flatness.append(count_flatness)
        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                mol_vec = mol_vec.sum(dim=0) / a_size
                #mol_vec = mol_vec.sum(dim=0)
                if self.args.Tmelt:
                    #mol_vec.add(symm[i])
                    ring = float(sssr[i])
                    mol_vec = torch.cat([mol_vec, torch.FloatTensor([ring])])
                    mol_vec = torch.cat([mol_vec, torch.FloatTensor([float(flatness[i])])])
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        
        if self.use_input_features:
            features_batch = features_batch.to(mol_vecs)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
            mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class MPNEncoder_attention(nn.Module):
    """A message passing neural network for encoding a molecule and adding an attention later"""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPNEncoder_attention, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args

        if self.features_only:
            return

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    def forward(self,
                mol_graph_solvent: BatchMolGraph, mol_graph_solute: BatchMolGraph,
                features_batch: List[np.ndarray] = None):
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

            if self.features_only:
                return features_batch

        molvecs_solvent, atom_hiddens_solvent, a_scope_solvent = self.message_passing(mol_graph_solvent)
        molvecs_solute, atom_hiddens_solute, a_scope_solute = self.message_passing(mol_graph_solute)

        return molvecs_solvent, molvecs_solute, \
               atom_hiddens_solvent, atom_hiddens_solute, a_scope_solvent, a_scope_solute  # num_molecules x hidden

    def message_passing(self,
                        mol_graph: BatchMolGraph):

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()

        if self.atom_messages:
            a2a = mol_graph.get_a2a()
        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

            if self.atom_messages:
                a2a = a2a.cuda()

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds),
                                        dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs, atom_hiddens, a_scope


class Attention(nn.Module):
    """A determination of attention coefficients"""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

         :param args: Arguments.
         :param atom_fdim: Atom features dimension.
         :param bond_fdim: Bond features dimension.
         """
        super(Attention, self).__init__()
        self.hidden_size = args.hidden_size
        self.atom_messages = args.atom_messages
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.layers_per_message = 1
        self.args = args
        self.dropout =  args.dropout

        # Activation
        self.sigm = nn.Sigmoid()
        self.act_func = nn.ReLU()

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(2 * self.hidden_size), requires_grad=False)

        # Input for attention layer
        input_dim_att = 2 * self.hidden_size

        self.att_1 = nn.Linear(input_dim_att, 2 * self.hidden_size)
        self.att_2 = nn.Linear(2 * self.hidden_size, 1)
        self.att_3 = nn.Softmax(dim=0)

        attnn = [
            self.dropout,
            nn.Linear(input_dim_att, 2 * self.hidden_size)
        ]
        attnn.extend([
             self.act_func,
             self.dropout,
             nn.Linear(2 * self.hidden_size, 1),
            ])
        attnn.extend([
            self.dropout,
            nn.Softmax(dim=0),
        ])
        self.attnn = nn.Sequential(*attnn)

    def forward(self, atom_solvents, atom_solute, a_scope_solvent, a_scope_solute):
        combined_vec = list()
        output_scope = list()
        start = 0
        for i in enumerate(a_scope_solvent):
            size = 0
            curr_solvents = atom_solvents.narrow(0, a_scope_solvent[i][0], a_scope_solvent[i][1])
            curr_solutes = atom_solute.narrow(0, a_scope_solute[i][0], a_scope_solute[i][1])
            size = a_scope_solvent[i][1]*a_scope_solute[i][1]
            output_scope.append((start, size))
            start = start + size
            for j in range(0, len(curr_solvents)):
                for k in range(0, len(curr_solutes)):
                    #temp = torch.cat([curr_solvents[j], curr_solutes[k]], dim=0).tolist()
            #temp = torch.FloatTensor(temp)
            #att_coeff = self.attnn(temp)
                    combined_vec.append(torch.cat([curr_solvents[j], curr_solutes[k]], dim=0).tolist())
        combined_vec = torch.FloatTensor(combined_vec)

        att_coeff = self.attnn(combined_vec)

        #att_coeff = self.att_3(att_coeff)
        #att_coeff = self.att(combined_vec)
        #print(att_coeff)
        #att_coeff = self.act_func(att_coeff)
        combined_vec = att_coeff*combined_vec

        return combined_vec, output_scope


class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args)

        output = self.encoder.forward(batch, features_batch)


        return output


class MPN_solvation(nn.Module):
    """A message passing neural network for encoding a set of molecules."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(MPN_solvation, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.encoder_solute = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)
        self.encoder_solvent = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self,
                batch: Union[List[List[str]], BatchMolGraph, BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input:  # if features only, batch won't even be used
            batch_solvent, batch_solute = mol2graph_solvation(batch, self.args)

        output_solute = self.encoder_solute.forward(batch_solute, features_batch)
        output_solvent = self.encoder_solvent.forward(batch_solvent, features_batch)

        output = torch.cat([output_solvent, output_solute], dim=1)

        return output


class MPN_solvation_attention(nn.Module):
    """A message passing neural network for encoding a set of molecules and including an attention layer."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(MPN_solvation_attention, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.encoder = MPNEncoder_attention(self.args, self.atom_fdim, self.bond_fdim)
        self.attention = Attention(self.args, self.atom_fdim, self.bond_fdim)
        self.use_input_features = args.use_input_features

    def forward(self,
                batch: Union[List[List[str]], BatchMolGraph, BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input:  # if features only, batch won't even be used
            batch_solvent, batch_solute = mol2graph_solvation(batch, self.args)
        mol_solvent, mol_solute, atom_hiddens_solvent, atom_hiddens_solute, a_scope_solvent, a_scope_solute\
            = self.encoder.forward(batch_solvent, batch_solute, features_batch)
        output, output_scope \
            = self.attention(atom_hiddens_solvent, atom_hiddens_solute, a_scope_solvent, a_scope_solute)

        # Readout
        mol_combs = []
        for i, (a_start, a_size) in enumerate(output_scope):
            if a_size == 0:
                mol_combs.append(self.cached_zero_vector)
            else:
                cur_hiddens = output.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0)/ a_size
                mol_combs.append(mol_vec)

        mol_combs = torch.stack(mol_combs, dim=0)  # (num_molecules, hidden_size)
        mol_vec_system = torch.cat([mol_solvent, mol_combs, mol_solute], dim=1)
        if self.use_input_features:
            features_batch = features_batch.to(mol_combs)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
            mol_vecs = torch.cat([mol_combs, features_batch], dim=1)  # (num_molecules, hidden_size)

        return mol_vec_system
