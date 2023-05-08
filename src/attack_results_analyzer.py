import numpy as np
import torch as torch
from pathlib import Path
from adv_attack import AdversarialExamplesSummary
import preprocess.resource_io as rio


importer = rio.ResourceImporter()

# result_dir = Path(__file__).parent.parent / "data" / "attack_results_f48_02"
# result_files = sorted(result_dir.glob("*"))
# result = importer.import_pickle_to_object(path=result_files[0])

result_file = Path(__file__).parent.parent / "data" / "k0.0-l10.15-lr0.1" \
                                                      "-ma100-ms1-2023-05" \
                                                      "-08_12:50:56.385216" \
                                                      ".pickle"

result = importer.import_pickle_to_object(result_file)

perts_abs = torch.abs(result.perturbations)

orig_zeros_indices = np.where(result.orig_labels == 0)[0]
orig_ones_indices = np.where(result.orig_labels == 1)[0]

num_oz_perts = len(orig_zeros_indices)
num_zo_perts = len(orig_ones_indices)

one_zero_perts_abs = perts_abs[orig_ones_indices, :, :]
zero_one_perts_abs = perts_abs[orig_zeros_indices, :, :]

GMP_ij_oz = torch.max(one_zero_perts_abs, dim=0).values
GMP_ij_zo = torch.max(zero_one_perts_abs, dim=0).values

GAP_ij_oz = torch.sum(one_zero_perts_abs, dim=0) / num_oz_perts
GAP_ij_zo = torch.sum(zero_one_perts_abs, dim=0) / num_zo_perts

GPP_ij_oz = torch.norm(one_zero_perts_abs, p=1, dim=0) / num_oz_perts
GPP_ij_zo = torch.norm(zero_one_perts_abs, p=1, dim=0) / num_zo_perts

S_ij_oz = GMP_ij_oz * GPP_ij_oz
S_ij_zo = GMP_ij_zo * GPP_ij_zo

S_j_oz = torch.sum(S_ij_oz, dim=1)
S_j_zo = torch.sum(S_ij_zo, dim=1)
