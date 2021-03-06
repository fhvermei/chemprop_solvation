from chemprop_solvation.solvation_estimator import load_DirectML_Gsolv_estimator, \
    load_DirectML_Hsolv_estimator, load_SoluteML_estimator

#####################################################
# Example 1. DirectML_Gsolv estimator

# Load DirectML_Gsolv estimator and do sample calculations
Gsolv_estimator = load_DirectML_Gsolv_estimator()
smiles = [['CC1=NC=CC=C1', 'CCCCCCCC'], # solvent, solute
         ['CC(CC(C)C)=O', 'Oc1ccccc1']] # solvent, solute
avg_pre, epi_unc, valid_indices = Gsolv_estimator(smiles)
# Output1. avg_pre
# It is the average solvation free energy prediction for each solvent-solute pair in kcal/mol.
# Expected outcome: avg_pre = [-4.791212034620899, -9.37750914598095]
# Output2. epi_inc
# It is the variance on predictions (=epistemic or model uncertainty) for each solvent-solute pair in kcal/mol.
# Expected outcome: epi_inc = [0.0015548871707848173, 0.000883700125432365]
# Output3. valid_indices
# It is a list of valid SMILES indices that could be used for the prediction.
# Expected outcome: valid_indices = [0, 1]

#####################################################
# Example 2. DirectML_Hsolv estimator

# Load DirectML_Gsolv estimator and do sample calculations
Hsolv_estimator = load_DirectML_Hsolv_estimator()
smiles = [['c1ccccc1', 'Cn1cccc1'], # solvent, solute
         ['C[S](C)=O', 'CCCC']] # solvent, solute
avg_pre, epi_unc, valid_indices = Hsolv_estimator(smiles)
# Output1. avg_pre
# It is the average solvation enthalpy prediction for each solvent-solute pair in kcal/mol.
# Expected outcome: avg_pre = [-9.508695074914133, -2.8130532493809155]
# Output2. epi_inc
# It is the variance on predictions (=epistemic or model uncertainty) for each solvent-solute pair in kcal/mol.
# Expected outcome: epi_inc = [0.0474315517261941, 0.055200657969424735]
# Output3. valid_indices
# It is a list of valid SMILES indices that could be used for the prediction.
# Expected outcome: valid_indices = [0, 1]

#####################################################
# Example 3. SoluteML estimator

# Load SoluteML estimator and do sample calculations
SoluteML_estimator = load_SoluteML_estimator()
smiles = [['CCCCCCCC'],  # solute
          ['Cc1ccccc1']] # solute
avg_pre, epi_unc, valid_indices = SoluteML_estimator(smiles)
# Output1. avg_pre
# It is the average solute parameter (E, S, A, B, L) prediction for each solute.
# Expected outcome:
# avg_pre = [[0.0033575539182320426, 0.0002618650031903691, 0.0012404804461126284, -3.3407873290287605e-05, 3.6623056859897627],
# [0.6076908247229844, 0.5312452682796652, 0.001725445237505657, 0.14568006907274622, 3.3648599797284917]]
# Output2. epi_inc
# It is the variance on predictions (=epistemic or model uncertainty) for each solute.
# Expected outcome:
# epi_inc = [[0.0001796549953840305, 0.00014738685559645396, 5.848635942567972e-05, 0.00016496458160710926, 0.00331170315682501],
# [0.00023044562651108525, 0.0003164094502622555, 4.004680160551078e-05, 0.0001829383390569367, 0.003946935292885148]]
# Output3. valid_indices
# It is a list of valid SMILES indices that could be used for the prediction.
# Expected outcome: valid_indices = [0, 1]
