# QMex-ILMM
"Extrapolative prediction of small-data molecular property using quantum mechanics-assisted machine learning" (to be submitted)

Hajime Shimakawa, Akiko Kumada, and Masahiro Sato

Department of Electrical Engineering & Information Systems, School of Engineering, 
The University of Tokyo, 7-3-1 Hongo, Bunkyo-ku, 113-8656, Tokyo, Japan

## Requirements
- python 3.7
- rdkit 2020.09.1.0
- sklearn 1.0.2
- openbabel 2.4.1
- pymatgen 2022.0.14
## Content
- ```data_test/```: data for QMex-ILR test (whole data are available in Zenodo https://doi.org/10.5281/zenodo.8177233)
- ```chemical_category.py```: implementation of chemical category from SMILES.
- ```get_protonated_smiles.py```: code to determine SMILES that can be protonated or de-protonated for chemical category [1].
- ```ILR_test.ipynb```: jupyternotebook for test of prediction with QMex-ILR.
## Reference
We use the public code by J. Wu et al.

[1] Wu, J., Wan, Y., Wu, Z., Zhang, S., Cao, D., Hsieh, C. Y., & Hou, T. (2022). MF-SuP-pKa: multi-fidelity modeling with subgraph pooling mechanism for pKa prediction. Acta Pharmaceutica Sinica B.
