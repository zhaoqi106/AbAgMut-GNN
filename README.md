# NEW3 Quick Start

NEW3 is a PyTorch/PyTorch Geometric project for antibody–antigen complex mutation ΔΔG prediction.

## Environment

Tested environment:

```text
python: 3.10.13
torch: 2.5.1
torch_geometric: 2.7.0
pandas: 2.3.3
numpy: 1.26.4
Bio / Biopython: 1.86
esm: 2.0.0
antiberty: installed, no __version__ field
torch cuda: 12.1
```

Additional external tools required for full preprocessing:

```bash
mkdssp
FoldX
ESM2 local weight: /home/zhao/esm2_t33_650M_UR50D.pt
```

## Project Path

The default project root is set in `config.py`:

```text
/home/zhao/gwc/NEW3
```

If the project is placed elsewhere, modify:

```python
PROJECT_ROOT = Path("/home/zhao/gwc/NEW3")
```

## Required Data Layout

For the default `SKEMPI` dataset:

```text
/home/zhao/gwc/NEW3/
├── data/SKEMPI/
│   ├── SKEMPI.csv
│   ├── wt/                 # wild-type PDB files: {PDB}.pdb
│   ├── mut/                # mutant PDB files: {ID}.pdb
│   └── complexsplits/      # fold_1 ... fold_5
└── feature/SKEMPI/
    ├── dssp/wt, dssp/mut
    ├── antiberty/wt, antiberty/mut
    ├── esm/wt, esm/mut
    └── graph/              # graph cache: {ID}.pt
```

The CSV should contain at least:

```text
ID,PDB,Mutation,Partners,ddG
```

Example `Partners` format:

```text
HL_A
HL_AB
AB_CD
```

The left side of `_` is treated as antibody chains, and the right side is treated as antigen chains.

## Full Run from Raw PDB Files

Run the following commands in order:

```bash
cd /home/zhao/gwc/NEW3

# 1. Generate complex-cluster five-fold split, if not already available
python complex-clustersplit_cv5.py

# 2. Generate DSSP features
python dssp.py

# 3. Generate antibody-side AntiBERTy features
python AntiBERTy.py

# 4. Generate antigen-side ESM2 features
python ESM2.py

# 5. Build graph cache files: feature/SKEMPI/graph/{ID}.pt
python preprocess.py

# 6. Train the model
python train.py
```

## Minimal Run If Graph Cache Already Exists

If `feature/SKEMPI/graph/{ID}.pt` files already exist, you can skip preprocessing and run only:

```bash
cd /home/zhao/gwc/NEW3
python train.py
```

## Output

Training outputs are saved under:

```text
output/SKEMPI/full_model/
├── checkpoints/
├── logs/
└── predict/
```

Current checkpoint output:

```text
fold_01_best_composite.pt
fold_02_best_composite.pt
...
fold_05_best_composite.pt
```

This version does not save `best_rmse` or `best_pearson` checkpoints.

## External Validation

Example:

```bash
cd /home/zhao/gwc/NEW3

python val.py \
  --project-root /home/zhao/gwc/NEW3 \
  --model-dataset SKEMPI \
  --val-dataset SARS \
  --ablation-tag full_model \
  --ckpt-kind best_composite
```

The validation dataset should have its own CSV and graph cache files, for example:

```text
data/SARS/SARS_val.csv
feature/SARS/graph/{ID}.pt
```

## Common Checks

```bash
# Check code files
ls config.py dataset.py pdb_graph.py preprocess.py model.py train.py val.py

# Check data
ls data/SKEMPI/SKEMPI.csv
ls data/SKEMPI/wt | head
ls data/SKEMPI/mut | head

# Check graph cache
find feature/SKEMPI/graph -name "*.pt" | head

# Check external tools
which mkdssp
ls -lh /home/zhao/esm2_t33_650M_UR50D.pt
ls -lh /home/zhao/gwc/MODEL/FOLDX/foldx
```
