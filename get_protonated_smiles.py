#!/usr/bin/env python
# coding: utf-8
from __future__ import division
from __future__ import unicode_literals
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd
import os, sys
# from https://github.com/wujialu/MF-SuP-pKa
smarts_file = "./data/smarts_pattern_ionized.txt"

def split_acid_base_pattern(smarts_file):
    df_smarts = pd.read_csv(smarts_file, sep="\t")
    df_smarts_acid = df_smarts[df_smarts.AB == "A"]
    df_smarts_base = df_smarts[df_smarts.AB == "B"]
    return df_smarts_acid, df_smarts_base

def unique_acid_match(matches):
    single_matches = list(set([m[0] for m in matches if len(m)==1]))
    double_matches = [m for m in matches if len(m)==2]
    single_matches = [[j] for j in single_matches]
    double_matches.extend(single_matches)
    return double_matches

def match_acid(df_smarts_acid, mol, get_substructure_type=False):
    matches = []
    substructure_types = []
    for idx, name, smarts, index, _, acid_base in df_smarts_acid.itertuples():  # name is the substructure id
        pattern = Chem.MolFromSmarts(smarts)
        match = mol.GetSubstructMatches(pattern)
        if len(match) == 0:
            continue
        if len(index) > 2:
            index = index.split(",")
            index = [int(i) for i in index]
            for m in match:
                matches.append([m[index[0]], m[index[1]]])
            substructure_types.append(name)
        else:
            index = int(index)
            for m in match:
                matches.append([m[index]])
            substructure_types.append(name)
    matches = unique_acid_match(matches)
    matches_modify = []
    for i in matches:
        for j in i:
            matches_modify.append(j)
    if get_substructure_type:
        return list(set(matches_modify)), list(set(substructure_types))
    else:
        return list(set(matches_modify))

def match_base(df_smarts_base, mol, get_substructure_type=False):
    matches = []
    substructure_types = []
    for idx, name, smarts, index, _, acid_base in df_smarts_base.itertuples():
        pattern = Chem.MolFromSmarts(smarts)
        match = mol.GetSubstructMatches(pattern)
        if len(match) == 0:
            continue
        if len(index) > 2:
            index = index.split(",")
            index = [int(i) for i in index]
            for m in match:
                matches.append([m[index[0]], m[index[1]]])
            substructure_types.append(name)
        else:
            index = int(index)
            for m in match:
                matches.append([m[index]])
            substructure_types.append(name)
    matches = unique_acid_match(matches)
    matches_modify = []
    for i in matches:
        for j in i:
            matches_modify.append(j)
    if get_substructure_type:
        return list(set(matches_modify)), list(set(substructure_types))
    else:
        return list(set(matches_modify))


def get_ionization_aid(mol, acid_or_base=None, get_substructure_type=False):
    df_smarts_acid, df_smarts_base = split_acid_base_pattern(smarts_file)

    if mol == None:
        raise RuntimeError("read mol error")
    if get_substructure_type:
        acid_matches, acid_type = match_acid(df_smarts_acid, mol, get_substructure_type)
        base_matches, base_type = match_base(df_smarts_base, mol, get_substructure_type)
        if acid_or_base == None:
            return acid_matches, base_matches, acid_type, base_type
        elif acid_or_base == "acid":
            return acid_matches, acid_type
        else:
            return base_matches, base_type
    else:
        acid_matches = match_acid(df_smarts_acid, mol)
        base_matches = match_base(df_smarts_base, mol)
        if acid_or_base == None:
            return acid_matches, base_matches
        elif acid_or_base == "acid":
            return acid_matches
        else:
            return base_matches

def calculate_num_sites(smiles):
    num_acid = []
    num_base = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        mol = AllChem.AddHs(mol)
        acid_idx, base_idx = get_ionization_aid(mol)
        num_acid.append(len(acid_idx))
        num_base.append(len(base_idx))
    return num_acid, num_base

def get_acid(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol_h = AllChem.AddHs(mol)
    acid_idx, base_idx = get_ionization_aid(mol_h)
    if len(acid_idx) == 0 and len(base_idx) == 1:
        atom = mol.GetAtomWithIdx(base_idx[0])
        atom.SetFormalCharge(atom.GetFormalCharge() + 1)
        atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
        acid = Chem.MolToSmiles(mol)
    else: acid = None
    return acid

def get_base(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol_h = AllChem.AddHs(mol)
    acid_idx, base_idx = get_ionization_aid(mol_h)
    if len(base_idx) == 0 and len(acid_idx) == 1:
        atom = mol.GetAtomWithIdx(acid_idx[0])
        atom.SetFormalCharge(atom.GetFormalCharge() - 1)
        atom.SetNumExplicitHs(atom.GetTotalNumHs() - 1)
        base = Chem.MolToSmiles(mol)
    else: base = None
    return base


def get_acid_base(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol_h = AllChem.AddHs(mol)
    acid_idx, base_idx = get_ionization_aid(mol_h)
    l_acid=[]
    l_base=[]
    for ib in base_idx:
        mol = Chem.MolFromSmiles(smiles)
        atom = mol.GetAtomWithIdx(ib)
        atom.SetFormalCharge(atom.GetFormalCharge() + 1)
        atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
        acid = Chem.MolToSmiles(mol)
        l_acid.append(acid)
    for ia in acid_idx:
        mol = Chem.MolFromSmiles(smiles)
        atom = mol.GetAtomWithIdx(ia)
        atom.SetFormalCharge(atom.GetFormalCharge() - 1)
        atom.SetNumExplicitHs(atom.GetTotalNumHs() - 1)
        base = Chem.MolToSmiles(mol)
        l_base.append(base)
    return list(set(l_acid)), list(set(l_base))



def categorize_acid_base(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol_h = AllChem.AddHs(mol)
    acid_idx, base_idx = get_ionization_aid(mol_h)
    is_acid=1 if len(acid_idx)>0 else 0
    is_base=1 if len(base_idx)>0 else 0
    return is_acid, is_base


def get_acid_base_list(l_smiles):
    l_acid = []
    l_base = []
    l_num_a = []
    l_num_b = []
    for s in l_smiles:
        l_a,l_b = get_acid_base(s)
        l_acid.append(l_a)
        l_base.append(l_b)
        l_num_a.append(len(l_a))
        l_num_b.append(len(l_b))
    return l_acid, l_base, l_num_a, l_num_b
