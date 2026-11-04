# Training

[`train.py`](../src/pixl/train.py) trains `EncoderPixl` ([model.md](model.md#model)) from random weights for one pass over the 1,000,000 train scenes of the `scenes` config ([data.md](data.md#scenes)): 31,250 steps of 32.

```bash
uv run python -m pixl.train          # the full run
uv run python -m pixl.train --quick  # 40 steps on 2,000 scenes: checks every step works
uv run python -m pixl.train --help   # every setting
```

The [Docker image](server.md) runs the same command on a GPU server, which is where full runs belong: one pass took about a day on the measured Apple-silicon GPU, and training on the Mac's GPU also made the terminal stop responding.

## Outputs

Training loads the scenes from the Hub (about 1 GB, downloaded and encoded once into the Hugging Face cache) and writes `checkpoints/<run>/`:

| File | Contents |
|---|---|
| `checkpoint.pt` | Everything needed to resume, and the run's log, preview and final scores ([below](#checkpoints-and-resuming)) |
| `samples-<step>.png` | At each evaluation: real preview scenes above the model's samples of their captions |

With `--quick`, the folder is `<run>-quick`, evaluation runs every 20 steps, warm-up takes 10 steps, and scores use every 50th evaluation scene, including the final scores.

## Devices and light local checks

`--device auto` (the default) picks CUDA, then Apple's GPU (MPS), then the CPU; `cuda`, `mps` and `cpu` force one. NVIDIA GPUs train in bf16; elsewhere training runs in float32 with `Config.recompute`, which batch 32 needs on a 24 GB Mac. On the measured Mac a step took about 2.7 seconds on the GPU, and 4–6 seconds on 4 CPU threads.

A check on a laptop is lighter on the CPU, at low priority and on a few threads, which leaves the GPU and most cores to everything else:

```bash
OMP_NUM_THREADS=4 nice -n 19 uv run python -m pixl.train --quick --device cpu
```

## Notebooks

Open them in VS Code with the project's `.venv` kernel, or start Jupyter with `uv run --with jupyter jupyter lab`. Their checkpoint paths are relative to `notebooks/`, so use that as the kernel's working directory.

| Notebook | Does |
|---|---|
| [`train.ipynb`](../notebooks/train.ipynb) | Runs `train()`, then shows a checkpoint's curves, scores and samples, including a run trained elsewhere |
| [`generate.ipynb`](../notebooks/generate.ipynb) | Loads a checkpoint with `load_model()` and draws scenes from the preview captions and your own |

Both default to the CPU at low priority (`DEVICE = "cpu"`, 4 threads), and the train notebook to the 40-step check (`QUICK = True`).

## Measures

The evaluation pool contains 500 train scenes and all `new sprites` and `new pair` test scenes. Every 2,000 steps, a full run measures every fourth entry in that pool, and at the end every entry:

| Measure | Question it answers |
|---|---|
| Loss | How well the scenes are modelled, per pixel |
| Position | Does a scene's caption beat the same caption with two objects' quadrants exchanged? (chance 0.5) |

The `swapped` captions in the dataset always exchange the first two objects in the caption.

## Position margin

Next to the pixel loss, each step scores every kept caption a second time with two random objects' places exchanged (`at the top left` and so on; every scene caption has one per object). A hinge term adds `max(0, margin - gap)`, where `gap` is how many nats per scene the swapped caption's loss exceeds the true one's. It is averaged over the batch with dropped captions counting 0, and weighted like the pixel loss per scene. Only reading the places can open that gap. `--margin` sets it (2 nats by default, a guess, not a tuned value), and `--margin 0` turns it off and trains exactly as before it existed.

The extra rows nearly double each step's work and activation memory. `batch loss` in the log is the true captions' pixel loss only.

`new pair` position well above 0.5 means the model reads positions; `train` high with `new pair` near 0.5 means it learned the object combinations it saw. Expect the `new sprites` loss to stay high: the run teaches the train sprites' shapes, not new ones.

## Checkpoints and resuming

A restarted run resumes from its last saved evaluation. Each save writes `checkpoint.tmp` and then renames it to `checkpoint.pt`, so a run stopped while saving keeps its previous checkpoint. The sample order is drawn once from a fixed seed, caption dropout and the margin's place swaps draw from their own seeded `random.Random`, and preview sampling runs on forked PyTorch random states without drawing from it. The checkpoint holds:

| Key | Contents |
|---|---|
| `model`, `optimizer`, `config`, `step` | Weights, optimizer state, model shape and steps done |
| `random` | States of the caption-dropout and place-swap generator and of PyTorch's CPU and GPU generators |
| `settings` | The command-line settings except `device` and `checkpoints` |
| `fingerprints` | Short SHA-256s of train/test tokens, captions, swapped captions and test kinds, plus `tokenizer.json` and `palette.json` |
| `log`, `preview`, `final` | Measures at each evaluation, the preview captions and scenes, and the final scores (subsampled in quick mode) |

A restarted run with the same `--run` resumes from the checkpoint, and stops instead if its saved settings or fingerprints differ: change `--run` to train with other settings. The checkpoint directory and device may change when resuming, though changing hardware or software can change numerical results. A recorded 40-step smoke test matched an uninterrupted run to within 4e-7 after resuming at step 20.

## Background

Earlier runs fine-tuned the decoder-only model, with captions from frozen CLIP or the body of Image GPT, on one- and two-object scenes; position stayed at chance (0.50–0.54). In a two-object scene, knowing the order is worth only ln 2 nats out of about 240, so the loss gave position almost no gradient. Four objects in random caption order raise that to ln 24 ≈ 3.2 nats, but that was not enough either. After one pass without the margin (run `scenes-four`), the true caption cost 14.6 nats per scene less than an empty one on new pairs (16.5 against 31.1), yet the same as the swapped caption to two decimals, and position stayed at 0.48–0.49. The model read which objects to draw but not where. Hence the [position margin](#position-margin). One pass with `--margin 2` (run `scenes-four-margin`) ended at 0.80 position on all 1,000 new-pair scenes (0.82 train, 0.60 new sprites), against 0.50 without it, with the same losses (0.016 train and new pair). Position stayed near chance until about step 24,000, jumped to 0.70 by step 26,000, and was still rising at the last step. Two passes (run `scenes-four-margin-2ep`, `--epochs 2`, about 5 hours) reached 1.00 position on all train and new-pair scenes (0.86 new sprites), and the loss fell to 0.006 train and 0.005 new pair. Position again jumped between steps 22,000 and 26,000, then reached 1.00 at step 32,000 and stayed there. At 1.00 the measure is saturated: it no longer tells better models apart, and it scores captions against real scenes rather than checking where generated sprites land.
