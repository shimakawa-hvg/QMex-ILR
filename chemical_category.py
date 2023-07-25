import pandas as pd
import numpy as np
import pybel, os, sys
from sklearn import preprocessing
from pymatgen.io.gaussian import GaussianInput
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AtomPairs import Pairs
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
from contextlib import redirect_stdout

import get_protonated_smiles
from pymatgen.core.periodic_table import Element
d_ele = {str(Element.from_Z(i)):i for i in range(1,119)}
l_ele = list(d_ele.keys())
l_non_metal = [ek for ek,ev in d_ele.items() if ev in [1,2,5,6,7,8,9,10,14,15,16,17,18,33,34,36,35,36,52,53,54,85,86]]
l_metal = list(set(l_ele)-set(l_non_metal))
l_argon = ['He','Ne','Ar','Kr','Xe','Rn']
l_halogen = ['F','Cl','Br','I','At']
l_atom = ['H','B','C','N','O','F','Si','P','S','Cl','Br','I']
l_other = list(set(l_non_metal)-set(l_atom)-set(l_argon))
d_bond = {
'SINGLE':Chem.rdchem.BondType.SINGLE,
'DOUBLE':Chem.rdchem.BondType.DOUBLE,
'TRIPLE':Chem.rdchem.BondType.TRIPLE,
'AROMATIC':Chem.rdchem.BondType.AROMATIC,
}
d_sub = {
'O_single':'*O*',
'O_double':'*=O',
'N_triple':'*#N',
'amine':'[NX3;H2,H1,H0;!$(NC=[!#6])]',
'alcohol':'[OX2H]([#6X4])', # Must be connected to sp3
'ether':'[OD2]([#6X4])[#6X4]', # Must also be connected to sp3
'cyano':'C#N',
'carboxylic':'[#6]([CX3](=O)[OX2H1])', # Must be connected to another carbon, removes the possibility of carbamates or similar groups.
'carbonyl':'[$([CX3]=[OX1]),$([CX3+]-[OX1-])]',
'amide':'[NX3][CX3](=[OX1])[#6]'}

def s2m(smi, AddHs=False):
  mol = Chem.MolFromSmiles(smi)
  if AddHs: return Chem.AddHs(mol)
  else: return mol

def m2s(mol):
  return Chem.MolToSmiles(mol)

def smiles2category(smiles, in_smi=False, in_pg=False):
  d_result = {}
  try: 
    mol = s2m(smiles, AddHs=True)
    smiles = m2s(mol)
  except: return d_result
  if in_smi: 
    d_result['smiles'] = smiles
  if in_pg: 
    d_result['pg'] = smiles2pg(smiles)
    d_result['polar'] = polar(pg=d_result['pg'])
  d_result['acid'], d_result['base'] = acid_base(smiles)
  d_result.update(atom_contain(mol, l_atom))
  d_result.update(bond_contain(mol, d_bond))
  d_result['pi'] = pi(mol)
  d_result['conjugated'] = conjugated(mol)
  d_result['ring'] = ring(mol)
  d_result['branch'] = branch(mol)
  d_result['rot'] = rot(mol)
  d_result['spiro'] = spiro(mol)
  d_result['hba'] = hba(mol)
  d_result['hbd'] = hbd(mol)
  d_result.update(sub_contain(mol, d_sub))
  return d_result

def atom_contain(mol, atoms):
  l_contain = [a.GetSymbol() for a in mol.GetAtoms()]
  l_charge = [a.GetFormalCharge() for a in mol.GetAtoms()]
  d_result = {}
  for a in atoms:
    if a in l_contain: d_result['with_'+a] = 1
    else: d_result['with_'+a] = 0
  # argon, halogen, metal
  d_result['argon'] = 1 if any([e in l_contain for e in l_argon]) else 0 
  d_result[f'halogen'] = 1 if any([e in l_contain for e in l_halogen]) else 0
  d_result['metal'] = 1 if any([e in l_contain for e in l_metal]) else 0
  d_result['other'] = 1 if any([e in l_contain for e in l_other]) else 0
  d_result['cation'] = 1 if sum(l_charge)>0 else 0
  d_result['anion'] = 1 if sum(l_charge)<0 else 0
  return d_result

def bond_contain(mol, d_bond):
  l_contain = [b.GetBondType() for b in mol.GetBonds()]
  d_result = {}
  for bk,bv in d_bond.items():
    if bv in l_contain: d_result[bk] = 1
    else: d_result[bk] = 0
  return d_result

def pi(mol):
  """
  https://future-chem.com/ap-dp-fingerprint/
  """
  fp = Pairs.GetAtomPairFingerprint(mol)
  explained = [Pairs.ExplainPairScore(k) for k in fp.GetNonzeroElements().keys()]
  n_pi = 0
  for type1,_,type2 in explained:
    n_pi += (type1[2]+type2[2])
  return 1 if bool(n_pi) else 0
def branch(mol):
  return 1 if bool(m2s(mol).count("(")) else 0
def conjugated(mol):
  return 1 if any([b.GetIsConjugated() for b in mol.GetBonds()]) else 0
def ring(mol):
  return 1 if bool(mol.GetRingInfo().NumRings()) else 0
def rot(mol):
  return 1 if bool(AllChem.CalcNumRotatableBonds(mol)) else 0
def spiro(mol):
  return 1 if bool(AllChem.CalcNumSpiroAtoms(mol)) else 0
def hba(mol):
  return 1 if bool(AllChem.CalcNumHBA(mol)) else 0
def hbd(mol):
  return 1 if bool(AllChem.CalcNumHBD(mol)) else 0

def sub_contain(mol, d_sub):
  d_result = {}
  for subk,subv in d_sub.items():
    match = Chem.MolFromSmarts(subv)
    if bool(mol.HasSubstructMatch(match)):
      d_result[subk] = 1
    else: d_result[subk] = 0
  return d_result

def smiles2pg(smiles):
  try:
    mymol = pybel.readstring("smi", smiles)
    mymol.make3D()
    gjf_line = '(Iso'.join([l.rstrip() for l in mymol.write('gjf').split('(Iso')])
    gjf_line = gjf_line.replace('Put Keywords Here, check Charge and Multiplicity.','p sp').replace('\n \n','\naaa\n')+'\n\n\n'
    gjf = GaussianInput.from_string(gjf_line)
    pg_analyzer = PointGroupAnalyzer(gjf.molecule)
    pg = pg_analyzer.sch_symbol
  except:
    pg = None
  return pg

def polar(pg):
  # https://en.wikipedia.org/wiki/Molecular_symmetry
  l_asym = [None,'C','C1','C2','C3','Cs','Ci','C*v','C2v','C3v','C4v','C5v','C6v']
  if pg in l_asym: return 1
  else: return 0

def acid_base(smiles):
  acid,base = get_protonated_smiles.categorize_acid_base(smiles)
  return acid,base

def make_category(smiles_list, in_smi=False, in_pg=False):
  l_result = []
  for s in smiles_list:
    d = smiles2category(s, in_smi, in_pg)
    l_result.append(d)
  df_cate = pd.DataFrame(l_result)
  return df_cate

def make_interaction_term(df_feature, df_category, degree=2, drop_rate=0.1, exclude_low_degree=False, is_new=False):
  """
  df_pf: result of PolynomialFeatures
  """
  degree = int(degree)
  df_cate = df_category.select_dtypes(include=[int,float])
  idx = df_feature.index
  idx_nan = df_cate[df_cate.isna().any(axis=1)].index
  df1 = pd.DataFrame(df_cate.values, index=idx, columns=[f"{c.replace('with_','')}1" for c in df_cate.columns])
  df0 = pd.DataFrame(abs(df_cate-1).values, index=idx, columns=[f"{c.replace('with_','')}0" for c in df_cate.columns])
  df01 = pd.concat([df1,df0],axis=1)
  if degree==-2:
    return df_feature, None
  elif degree==-1:
    df_pf=df_cate.copy()
  elif degree==0:
    df_pf=df01.copy()
  else:
    pf = preprocessing.PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False,)
    df_pf = pd.DataFrame(pf.fit_transform(np.nan_to_num(df01.values)), index=df01.index, columns=['*'.join(l.split(' ')) for l in pf.get_feature_names_out(df01.columns)])
    df_pf.loc[idx_nan] = np.nan

  if not is_new:
    df_pf = df_pf.drop(columns=df_pf.columns[((df_pf==0).sum()<=drop_rate*len(df_pf))|((df_pf!=0).sum()<=drop_rate*len(df_pf))])

  if exclude_low_degree:
    df_pf = df_pf[[c for c in df_pf.columns if len(str(c).split('*'))==degree]]
    df_result=pd.DataFrame([])
  else: 
    df_result = df_feature.copy()
  df_f = df_feature.copy()
  df_f['1'] = 1 # for intercept
  if degree>0:
    for f in df_f.columns:
      v = df_f[f].values.reshape(-1,1)*df_pf.values
      col = [f"{f}*{c}" for c in df_pf.columns]
      df_inter_tmp=pd.DataFrame(v, index=idx, columns=col)
      df_result = pd.concat([df_result,df_inter_tmp],axis=1)
  return df_result, df_pf

  