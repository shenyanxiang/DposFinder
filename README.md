# DposFinder

DposFinder is a transformer-based protein language model for phage-encoded depolymerase and their target serotype prediction.

## Setup

To install DposFinder, follow these steps：

1. Clone the repository:
```
git clone https://github.com/shenyanxiang/DposFinder
```


2. create a new vertual environment, for example：

```
cd DposFinder
conda env create -f environment.yml
conda activate DposFinder
```

3. Download trained DposFinder model from the link below and put it into `model` directory:

https://zenodo.org/records/13310759

or you can retrain DposFinder model with optional params by running:
```
python ./code/main.py [--FLAGS]
```
params can be checked by command:
```
python ./code/main.py -h
```

## Usage

Basic depolymerase prediction on a FASTA file:
```
python ./code/main.py --mode predict \
  --data_path ./data/ \
  --test_data test_set.fasta
```

Common optional arguments:
```
  --batch_size 32             # reduce this if you encounter CUDA OOM
  --return_attn               # save attention maps
  --return_subseq             # extract high-attention subsequence embeddings (used for serotype prediction)
  --return_subseq_len 350     # length of pooled subsequence window (default 350)
```

Help:
```
python ./code/main.py -h
```

Results are saved under the directory you provide via `--data_path` (default: `./data/`).  

## Serotype Prediction

DposFinder can infer the likely capsular / serotype of a predicted depolymerase by aligning an extracted high‑attention subsequence against reference depolymerase subsequences with known serotypes.

Reference subsequence FASTA files are provided (currently for Klebsiella pneumoniae and Acinetobacter baumannii) under:
```
data/subseq/
  Klebsiella_dpos_ref_350.fasta
  Klebsiella_dpos_ref_expand_350.fasta
  Acinetobacter_dpos_ref_350.fasta
```

you can also build your own reference data using
`
python ./code/main.py --mode predict\
  --data_path ./data/ \
  --test_data your_reference_fasta \
  --return_subseq \
  --return_subseq_len your_len
`

### 1. Prepare your depolymerase query sequence(s)

Put your depolymerase (or candidate) sequences in a FASTA file, e.g.:
```
example_depolymerase.fasta
```

### 2. Extract the subsequence used for serotype comparison

Run DposFinder in predict mode with subsequence extraction enabled:

```
python ./code/main.py --mode predict \
  --data_path ./data/ \
  --test_data example_depolymerase.fasta \
  --return_subseq \
  --return_subseq_len 350
```

This will generate a subsequence FASTA (high‑attention region) for each positive depolymerase.  
Depending on the current implementation, it may be written to a path like:
```
./data/subseq/example_depolymerase_subseq_350.fasta
```

### 3. Run serotype alignment-based prediction

Use the provided alignment script:

```
python ./code/serotype_predict.py \
  --reference_path ./data/subseq/Klebsiella_dpos_ref_expand_350.fasta \
  --query_path ./data/subseq/example_depolymerase_subseq_350.fasta \
  --output ./data/ \
  --k 3
```

### 4. Interpreting the results

The output table ( `./data/predicted_serotype.csv`) will list, per query sequence:
- query_id
- reference_id (best or top-k matched reference)
- normalized_score (alignment score / self-alignment score; 0–1)
- predicted_serotype (parsed from the reference description, typically the second token)

Higher normalized scores indicate closer similarity to that reference serotype-associated depolymerase subsequence. If multiple serotypes have similar scores, treat the result as ambiguous.

## Contact

Please contact Yanxiang shen at shenyanxiang@sjtu.edu.cn for questions.
