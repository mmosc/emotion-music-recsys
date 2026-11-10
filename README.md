# Measuring and Improving Emotion Calibration in Music Recommendations

Code to reproduce the experiments of the paper (IEEE Transactions on
Affective Computing). Datasets: ONION (auto-tagged GEMS), EMMA (human GEMS),
ONION-186 (ONION subsample with EMMA's shape).

## Setup

```bash
pip install -r requirements.txt
pip install -r requirements_torch.txt
pip install -r requirements_torch_geometric.txt
```

| path | content |
|---|---|
| `/opt/datasets/Music4All/music4all/onion_elliot/<dataset>/` | `interactions.tsv`, `splits/`, `{audio,textual,visual,emotion}_embeddings/` |
| `/opt/scratch/marta/results/` | all outputs (`RESULTS_BASE` in `utils.py`, configs) |

The scripts on GPU use `--gpus`; HPO shell scripts use `$PYTHON` (default `python`).

## 1. Datasets

Inputs (`data/`): Music4All-Onion `userid_trackid_timestamp.tsv`, feature files
`id_maest.tsv`, `id_bert.tsv`, `id_resnet.tsv`, `id_gems.tsv` (SRGNN-Emo), `id_emma.tsv`
(human GEMS).

1. `notebooks/period.ipynb` – year filter + binarization (>= 2 plays per user–track pair).
2. `notebooks/onion.ipynb` (March 2014) and `notebooks/emma_onion.ipynb` (EMMA tracks).
3. `notebooks/preprocess.ipynb` with `dataset = 'onion'` / `'emma'` – writes
   `interactions.tsv` and per-item `.npy` features; move the output to
   `onion_elliot/<dataset>/`.
4. ONION-186: `python build_onion186.py`
5. PCA-9 features (for the `pca9` configs), for each dataset and modality:
   ```bash
   python reduce_modality_features.py --modality audio \
       --input-dir .../onion_elliot/<dataset>/audio_embeddings/maest --n-components 9 \
       --output-dir .../onion_elliot/<dataset>/audio_embeddings/maest_pca9
   # likewise textual/bert -> bert_pca9, visual/resnet -> resnet_pca9
   ```
6. Fixed 10-fold splits: `python precompute_splits.py --datasets onion emma onion186`

## 2. Hyperparameter optimization (optional: results are in `config_files/tuned/`)

Grids are in `config_files/models/` (full features) and `config_files/pca9/`.

```bash
./run_hpo.sh --model itemknn  --gpus 0 --datasets onion emma onion186
./run_hpo.sh --model deepmf   --gpus 0 --datasets onion emma onion186
./run_hpo.sh --model lightgcn --gpus 0 --datasets onion emma onion186
for m in featureitemknn lightgcnm sibrar; do
  ./run_hpo.sh --model $m        --combos atv atve e --gpus 0 --datasets onion emma onion186
  ./run_hpo.sh --model $m --pca9 --combos atv atve e --gpus 0 --datasets onion emma onion186
done
# rerankers (need the trained backbones' recs from step 3)
./run_hpo_reranker.sh onion emma onion186 --backbone ItemKNN DeepMF LightGCN
./run_hpo_calibrated_reranker.sh --calibration-metric mahalanobis onion emma onion186 --backbone ItemKNN DeepMF LightGCN
./run_hpo_calibrated_reranker.sh --calibration-metric energy      onion emma onion186 --backbone ItemKNN DeepMF LightGCN
```

Tuned configs are written to `config_files/tuned/{full,pca9}/`.

## 3. 10-fold training and evaluation

Each run trains all 10 folds, then evaluates the saved recommendations with
`config_files/eval.yml` (nDCG, Mahalanobis `Calibration`, `EnergyCalibration` at
k = 1/5/10/20) into `.../performance/intermediate_fold_eval_<tag>_<fold>.json`.

```bash
# backbones, baselines and MMRS first ...
for c in config_files/tuned/full/{itemknn,deepmf,lightgcn,mostpop,featureitemknn,lightgcnm,sibrar}_*.yml \
         config_files/tuned/full/random_{onion,emma,onion186}.yml \
         config_files/tuned/pca9/*.yml; do
  python run_parallel_k_fold.py --config "${c%.yml}" --gpus 0
done
# ... then the rerankers, which read the backbones' recs
for c in config_files/tuned/full/{reranker,calibrated_reranker,random_shuffle}_*.yml; do
  python run_parallel_k_fold.py --config "${c%.yml}" --gpus 0
done
```

## 4. Emotion regression

Needs the trained LightGCN / DeepMF weights of step 3. For each dataset `d` in
`onion emma onion186`:

```bash
python run_regression_embeddings.py --dataset $d --model LightGCN --regressor linear
python run_regression_embeddings.py --dataset $d --model DeepMF   --regressor linear
python run_regression_streams.py    --dataset $d
python run_regression_mlp.py        --dataset $d --model LightGCN   # MLP grid search; likewise DeepMF, RawURM
```

Linear = least squares; MLP = one hidden ReLU layer, grid-searched by inner CV. Each run
writes per-fold r^2 (overall and per GEMS dimension, plus the train-mean baseline) to
`<results>/<dataset>/emotion_regression/`.
