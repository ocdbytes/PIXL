# Training

[`pixl.train`](../src/pixl/train.py) trains `EncoderPixl` from random initialization on the `scenes` dataset.

```bash
uv run python -m pixl.train          # full run
uv run python -m pixl.train --quick  # 40-step smoke test
uv run python -m pixl.train --help
```

The default run uses one million scenes for one epoch: 31,250 steps at batch size 32. Remote execution is documented in [remote-training.md](remote-training.md).

## Configuration

| Setting | Default | Description |
|---|---:|---|
| `--batch` | 32 | Scenes per optimizer step |
| `--epochs` | 1 | Passes over the training split |
| `--peak-lr` | `3e-4` | Learning rate after warm-up |
| `--min-lr` | `3e-5` | End of cosine decay |
| `--warmup` | 1,000 | Linear warm-up steps |
| `--weight-decay` | 0.1 | AdamW decay on matrices and embeddings |
| `--beta1`, `--beta2` | 0.9, 0.95 | AdamW moment decay |
| `--grad-clip` | 1.0 | Maximum gradient norm |
| `--caption-drop` | 0.1 | Empty-caption probability for classifier-free guidance |
| `--margin` | 2.0 | Required correct-versus-swapped gap in nats per scene |
| `--eval-every` | 2,000 | Steps between evaluation and checkpoint writes |
| `--seed` | 0 | Training random seed |

The learning rate increases linearly during warm-up and then follows cosine decay. All run settings except `--device` and `--checkpoints` are stored in the checkpoint and checked when resuming.

## Training step

```text
scene tokens + caption
        │
        ├── caption dropout
        │
        ├── pixel cross-entropy under the training caption
        │
        └── pixel cross-entropy under a position-swapped caption
                         │
                         ▼
              pixel loss + margin penalty
                         │
                         ▼
              backward pass → clip gradients → AdamW
```

Caption dropout replaces 10% of captions with the empty caption. These rows train the unconditional prediction used by classifier-free guidance. They contribute to pixel loss but not to the position margin.

CUDA training runs under bf16 autocast. CPU and MPS use float32 and enable gradient checkpointing through `Config.recompute`.

## Objective

Let \(P\) be the number of pixels and \(B\) the batch size. For scene \(i\), `model.loss()` returns the mean pixel cross-entropy \(l_i\) in nats per pixel.

For every non-empty caption, the same image is also scored under a caption with two object positions exchanged. If its mean loss is \(\tilde{l}_i\), the scene-level separation is

\[
g_i = P(\tilde{l}_i - l_i).
\]

The optimized batch objective is

\[
L = \frac{1}{B}\sum_i l_i
  + \frac{1}{BP}\sum_{i \in \text{non-empty}} \max(0, m - g_i),
\]

where \(m\) is `--margin`. With the default margin of 2, the swapped caption must cost at least two nats more per scene than the correct caption. A row contributes no margin penalty once that separation is reached. `--margin 0` disables swapped-caption scoring.

The additional forward rows nearly double compute and activation memory. The logged `batch loss` is only the first term: mean pixel loss under the training captions.

## Measures

The evaluation pool contains:

- 500 fixed training scenes;
- 1,000 `new sprites` scenes;
- 1,000 `new pair` scenes.

Intermediate evaluations score every fourth scene in this pool. The final evaluation scores all scenes.

| Measure | Definition |
|---|---|
| Loss | Mean pixel cross-entropy under the correct caption |
| Position | Fraction of scenes whose correct caption has lower loss than the dataset's position-swapped caption |

Position has a chance level of 0.5. `new pair` position tests whether the model uses location language for unseen combinations of known objects. `new sprites` also changes the underlying sprite images and is therefore a stronger distribution shift.

`batch loss` and evaluation loss are not directly comparable. Training loss includes dropped captions, averages changing model states between evaluations and uses bf16 on CUDA. Evaluation uses fixed scenes at one model state in float32.

## Reproducibility

`--seed` controls weight initialization, scene order, caption dropout and random position swaps. Evaluation-scene selection and preview generation use the separate fixed `EVAL_SEED`, so runs with different training seeds are evaluated on the same examples.

Use a distinct run name for each seed:

```bash
for seed in 0 1 2; do
  uv run python -m pixl.train \
    --run scenes-four-margin-2ep-s$seed --epochs 2 --seed $seed
done
```

The checkpoint stores Python and PyTorch random states. In the resume test, a 40-step interrupted run matched uninterrupted training within `4e-7` after resuming at step 20. Exact numerical agreement across different hardware or software versions is not guaranteed.

## Checkpoints

Each run writes to `checkpoints/<run>/`:

| Path | Contents |
|---|---|
| `checkpoint.pt` | Model, optimizer, step, configuration, random states, input fingerprints, logs and final scores |
| `samples-<step>.png` | Fixed real scenes and generated previews at an evaluation step |

Checkpoint writes are atomic: training writes `checkpoint.tmp` and then renames it to `checkpoint.pt`. An interrupted save therefore leaves the previous checkpoint intact.

Running the same `--run` resumes from its checkpoint. Resume stops if saved settings, model configuration or fingerprints of the dataset, tokenizer and palette differ. `recompute` may change because it follows the selected device. Use a new run name for any other configuration change.

## Quick mode and analysis

`--quick` trains for 40 steps on 2,000 scenes, evaluates every 20 steps and uses every 50th evaluation scene. Its output directory is `<run>-quick`.

For a low-impact CPU check:

```bash
OMP_NUM_THREADS=4 nice -n 19 uv run python -m pixl.train --quick --device cpu
```

[`notebooks/train.ipynb`](../notebooks/train.ipynb) plots checkpoint logs, scores and previews. [`notebooks/generate.ipynb`](../notebooks/generate.ipynb) loads a checkpoint for generation.
