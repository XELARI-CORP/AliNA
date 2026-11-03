# ELINA — MSA-aware RNA/DNA secondary-structure model

Describes branch `msa_na_model` (compared against `master`, the original AliNA).

ELINA takes a nucleic-acid sequence, optionally with an MSA of homologous sequences and an *input* secondary structure
(e.g. a coevolution-derived one, but dummy `"."*len(seq)` can be used too), and predicts for every nucleotide a probability distribution over its base-pairing partner.
It is used for two tasks (see [Tasks, losses, metrics](#tasks-losses-metrics)): **predicting** a structure and **correcting** the input structure.

## 1. What differs from `master`

| | `master` (AliNA) | `msa_na_model` (ELINA) |
|---|---|---|
| Model input | L×L tensor of pair tokens zero-padded to 256×256 | Per-nucleotide tokens `(msa, L)` (vocab 6) + input structure vector `(L)`; variable length, padded per batch |
| Architecture | U-Net: 4 conv encoder blocks → 3 transformer layers on 16×16 patches → 4 conv decoder blocks | 1D conv → MSA transformer → "complementary" graph layer → transformer encoders → bilinear pair head |
| Output | L×L pair-probability matrix (sigmoid) | `(L, L+1)` softmax along rows: partner distribution per nucleotide, column 0 = "unpaired" |
| Dataset | `AlinaDataset(nas, dimer_embeddings, with_adjacency)` → `(x, adjacency)`; `make_collate(max_len, center_pad)` | `AlinaDataset(nas, msa_*, inp/out_struct_source, valid)` → `AlinaDataPoint`; `collate_fn` → `AlinaBatch` |
| Checkpoint | `model_state_dict`, `model_params`, `dimer_embeddings`, `center_pad` | `model_state_dict` and `model_params` only (+ unwraps `torch.compile`); training checkpoints include `optim_state_dict` |
| Training code | none in the repo | `train_EliNA.py`, `train_utils.py`, `alina/utils.py` (losses, metrics, LR schedule, checkpointer), `configs/` |

## 2. Model (`alina/model/`)

`AliNA` (`alina/alina.py`) subclasses `Model` (`alina/model/model.py`); building blocks are in `modules.py`.

```
seq (b, msa, L) int   struct_vec (b, L) int
   │  seq: 0=pad A=1 U=2 G=3 C=4 N=5      struct_vec: partner index, -1 (unpaired/pad)
   ▼
Embedding                                   → (b, msa, L, dim)
Conv1d stack along L (k=3, SiLU between)    → (b, msa, L, dim)
MSATransformer × len(seq_encoders_order)    → (b, msa, L, dim)
   ├ row attention along L (RoPE)
   ├ column attention along msa (no RoPE)   ← skipped if skip_msa_att
   └ SwiGLU feed-forward
take MSA row 0 (the query) + sinusoidal PE  → (b, L, dim)
ComplementaryLayer(x, struct_vec)           → (b, L, dim)
   [x, is_free, is_paired] → Linear → + embedding of the partner position → SiLU → Dropout → Linear
prepend learnable "no-bound" vector         → (b, L+1, dim)
EncoderLayer × len(struct_encoders_order)   → (b, L+1, dim)   (pre-norm MHA + SwiGLU)
LayerNorm → x·W·xᵀ / √dim + bias, diagonal masked with -1e7 → (b, L+1, L+1)
   ├ row 0      → sigmoid → nbn (b, L+1)
   └ rows 1..L  → softmax → m   (b, L, L+1)
```

- **Output `m`:** `m[b,i,0]` = P(nucleotide *i* unpaired), `m[b,i,j+1]` = P(*i* pairs with *j*).
- **Layer sharing:** `*_encoders_order` lists layer indices to apply; repeated indices reuse the same weights. `[1,2,1,2]` = two distinct layers, applied four times. Indices must be 1..N with no gaps.
- **Masks:** padding tokens (`0`) are masked in row attention, padded MSA rows in column attention, and padded positions in the structure encoders.

Hyperparameters (`configs/hparams/elina_default.yaml`):

| key | default | meaning |
|---|---|---|
| `vocab_size` | 6 | pad + A, U, G, C, N |
| `dim` | 256 | model width |
| `conv_layers` | 2 | Conv1d layers (no activation after the last) |
| `seq_encoders_order` | `[1, 1]` | MSA-transformer layer order (one layer applied twice) |
| `struct_encoders_order` | `[1, 2, 1, 2]` | structure-encoder layer order |
| `heads` | 8 | attention heads |
| `convdrop`, `encdrop` | 0.15 | dropout in conv / encoder blocks |
| `skip_msa_att` | `True` | disables column (MSA-wise) attention, see [Known issues](#6-known-issues) #1 |

`conv_activation` (`SiLU`) and `norm_layer` (`LayerNorm`) are injected in `train_EliNA.py`, not in the config.

## 3. Dataset (`alina/dataset.py`)

`AlinaDataset(nas, msa_sample_fraction=1.0, msa_sample_max_size=100, cache=True, inp_struct_source="struct", out_struct_source=None, valid=False)`

- **Items:** each `nas[i]` is an `naskit` nucleic acid. `__getitem__` returns `AlinaDataPoint(seq (msa, L), inp_struct (L), out_struct (L), len)`.
- **Sequence / MSA:** row 0 is the sequence itself; `na.meta["msa"]` rows are appended if they are new and have the same length. Tokens: `A=1 U=2 G=3 C=4 N=5`, pad `0`.
- **Structures:** `inp_struct` is the model's input structure, `out_struct` is the training target. Each is a partner-index vector, `-1` = unpaired. A source is `None` (all `-1`), `"struct"` (the NA's own pairs) or a key of `na.meta` holding a dot-bracket string (e.g. a coevolution structure).
- **MSA sampling** (`make_msa_sample`): `k = clamp(ceil(msa_len · msa_sample_fraction), 1, msa_sample_max_size)`.
  - Training (`valid=False`): with p = 0.5 only the query row is used; otherwise the query plus `k-1` random other rows.
  - Validation (`valid=True`): the first `k` rows, deterministically.
- **Cache:** with `cache=True` items are stored compressed (int8/int16) after the first access. `save()`/`load()` pickle `nas`, the cache and the MSA/`valid` settings.
- **Collate** (`collate_fn` → `AlinaBatch`): `seq (b, max_msa, max_L)` zero-padded; `inp_struct`, `out_struct` `(b, max_L)` filled with `-1`; `lens` = true lengths. Max length is no longer fixed at 256 as in `master`.

## 4. Tasks, losses, metrics

`special_params.model_task` selects the task, and `train_utils.setup_train_modules` asserts the loss matches it:

| `model_task` | Meaning | Allowed `loss_fn` | Metrics (`alina/utils.py`) |
|---|---|---|---|
| `p` | **predict** the structure from sequence/MSA (input structure usually empty) | `CELoss` | `PredMetrics`: `recall`, `precision`, `Fscore` |
| `c` | **correct** an input (coevolution) structure | `MaskedCELoss`, `WeightedMaskedCELoss` | `ClasMetrics`: `preserved_tp_ratio`, `fixed_fp_ratio`, `Fscore` |

- **`CELoss`:** cross-entropy of `m` against `out_struct`, averaged over all `b·L` positions.
- **`MaskedCELoss`:** same, but only over positions where `inp_struct != -1` (the input pairs being judged).
- **`WeightedMaskedCELoss(pos_weight)`:** as `MaskedCELoss`, with weight `pos_weight` on positions whose true partner exists (`out_struct != -1`). Requires `special_params.loss_fn_params.pos_weight` to be set.
- **`LossFn(name, **params)`:** thin wrapper so the training loop always calls `loss_fn(pred, inp, target)`. Params come from `special_params.loss_fn_params`.
- **`PredMetrics`:** a position is correct if the arg-max partner equals the truth and its probability is ≥ 0.5. `recall` is correct pairs over true paired positions, `precision` is correct pairs over predicted paired positions, and `Fscore` is their harmonic mean.
- **`ClasMetrics`:**
  - `preserved_tp_ratio`: input pairs that are true and kept, with probability ≥ 0.5.
  - `fixed_fp_ratio`: input pairs that are wrong and were fixed or removed.
  - `Fscore`: their harmonic mean.
- **`Fscore`:** this is the number used to select the "best" checkpoint.

## 5. How to train

### Requirements
- Python ≥ 3.12 (the scripts use nested same-quote f-strings), an NVIDIA GPU and driver (`pynvml` is initialised on import of `alina.utils`; autocast/`GradScaler` are hard-coded to CUDA bf16).
- Packages (no pinned requirements file in the repo): `torch`, `naskit`, `hydra-core`, `omegaconf`, `mlflow`, `loguru`, `nvidia-ml-py` (`pynvml`), `numpy`, `tqdm`.
- A running MLflow server at `http://127.0.0.1:31420` (hard-coded in `train_EliNA.py:112`).

### Run
From the repo root (Hydra reads `configs/elina.yaml`):

```bash
python train_EliNA.py                                  # all defaults
python train_EliNA.py data=elina_pdb_updated_cls \
    special_params.model_task=c special_params.loss_fn=WeightedMaskedCELoss \
    special_params.loss_fn_params.pos_weight=3.0 special_params.save_by=_ \
    run_name_prefix=smoke_test const.DEVICE_IDX=0 const.MAX_TRAIN_STEPS=100 const.VALID_EVERY=10
python train_EliNA.py checkpoint_path=/path/to/step10000.pth lr_params=elina_tune   # fine-tune / resume
```

`CELoss` accepts no parameters, so keep `special_params.loss_fn_params` as `{}` for task `p`. The weight in the second example applies to `WeightedMaskedCELoss` only.

### Config layout (`configs/`)

`elina.yaml` composes six groups (override with `group=name`) plus top-level keys `run_name_prefix`, `work_dir`, `checkpoint_path`, `experiment_name`, `random_seed`.

| group | key parameters |
|---|---|
| `data/` | `train_data_path` (one `.pkl`); `valid_data_path` = `label\|path[:label\|path ...]`. `special_params.save_by` must equal one of the labels. |
| `hparams/` | model hyperparameters (section 2) |
| `const/` | `BATCH_SIZE 16`, `GRAD_ACUM 8` (effective batch 128), `WEIGHT_DECAY 0.01`, `MAX_TRAIN_STEPS 50000` (optimizer steps), `VALID_EVERY 250`, `SAVE_EVERY 10000` (`null` = off), `LOG_EVERY 15`, `CLIP_GRAD 0.1` (clip by value), `DEVICE_IDX 0` |
| `lr_params/` | `warmup`, `peak`, `c`, `min_lr`, `max_lr`, optional `rules: [[step, lr], ...]`. LR is linear warm-up to `peak`, then cosine decay to `min_lr` (speed set by `c`); `rules` force a constant LR from a given step. `elina_tune` has a lower peak for fine-tuning. |
| `special_params/` | `model_task` (`p`/`c`), `loss_fn`, `loss_fn_params`, `save_by`, `freeze_layers` (substrings of parameter names), `compile_model` (0/1) |
| `run_name_template/` | `elina_struct[1,2,1,2]_wd0.01_drop0.15` → cleaned to `elina_struct1212_wd0.01_drop0.15` |

`configs/alina*` and the `alina_*` files in each group are for the `master` branch and not used by `train_EliNA.py`.

### Expected input
- Pre-built, pre-cached `AlinaDataset` pickles (`AlinaDataset.load`), one for training and one or more labelled validation sets. Validation sets should have been created with `valid=True`. Building them is out of scope here.
- Per `na`: `seq`, optionally `meta["msa"]`, and the structure sources chosen when the dataset was built.
- **Optional checkpoint:** `checkpoint_path` pointing to a `.pth` produced by this trainer.

### What happens
Pre-validation on every valid set, then training with AdamW and bf16 autocast, with gradient accumulation and value clipping.
- Validation runs every `VALID_EVERY` optimizer steps.
- Training stops at `MAX_TRAIN_STEPS`.

### Output / artefacts
- **Checkpoints:** in `<work_dir>/<run_name>/`.

| file | when |
|---|---|
| `best_val{Fscore:.4f}_{label}_step{N}.pth` | new best `Fscore` on the `save_by` set (previous best is deleted) |
| `step{N}.pth` | every `SAVE_EVERY` steps |
| `finished.pth` | training reached `MAX_TRAIN_STEPS` |
| `stopped.pth` | Ctrl+C or SIGTERM |
| `error.pth` | any exception during training (then re-raised) |

  Each file holds `model_state_dict`, `model_params`, `optim_state_dict`. Loading uses `weights_only=False` (pickle), so only load trusted files.
- **MLflow:** experiment `experiment_name`, run `run_name`, whole config as params. Metrics are `Loss`, the task metrics, `Lr`, `step/s`, `GPUMem`, `GPUUtil` every `LOG_EVERY`, and `{label}_valid_Loss` / `{label}_valid_{metric}` every `VALID_EVERY`.
- **Hydra:** its own run folder under `outputs/` (gitignored) with the resolved config.

### Resume / fine-tune
`checkpoint_path` restores weights, `model_params` (architecture comes from the checkpoint; `hparams` are ignored) and the AdamW state. It does **not** restore the step counter, LR schedule, best-metric tracker or data order. Training restarts at step 0 with warm-up, so pick `lr_params` accordingly. `freeze_layers` freezes parameters whose names contain any listed substring.

## 6. Known issues

Ordered roughly by impact.

1. **The MSA is ignored with the default hparams.** With `skip_msa_att: True`, every MSA row is processed independently and only row 0 is used (`model.py:98`), so extra MSA sequences change nothing and only cost compute. Set `skip_msa_att: False` for the MSA to matter.
2. **Inference API is broken on this branch.** `AliNA.fold()` still unpacks a 4-tuple batch and calls `self(x)` (`alina/alina.py:128-130`), but the batch is now `AlinaBatch` and `forward` takes `(seq, struct_vec)` and returns a tuple. The bundled `alina/model/*.pth` files and `pretrained_model_parameters` use the old `master` architecture and no longer match `Model.__init__`. The README still documents the old API.
3. **Padding in the loss/head.** `CELoss` averages over padded positions too, so its value depends on batch padding. The pair head masks only the diagonal, not padded columns.
4. **Resume is partial and can overwrite files.** See [Resume / fine-tune](#resume--fine-tune). A resumed run with the same `run_name` writes `step{N}.pth` into the same directory and overwrites earlier files.
5. **Non-portable settings.** Hard-coded `sys.path` entries (`train_EliNA.py:2-3`), absolute `work_dir` and data paths (`/home/pavel/...`, `/mnt/HDD36/...`), MLflow URI.
6. **`AlinaDataset.load()` loses settings.** It does not restore `inp_struct_source`, `out_struct_source` or `cache`. These matter only for items that are not already cached. `__add__` merges in place and keeps only the left dataset's settings.
7. **Leftovers from `master`:** `alina/dictionary.py`, `pretrained_model_parameters`, and the `alina_*` configs (there is no ALINA training script).
8. `Model.pos_enc` and `RoPE.prepare_rope` use `@torch.compile`, so the first step is slower and needs a working compiler toolchain.
