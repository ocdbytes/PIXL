# Refine

`clip_filter.py` drops web pairs in `pixel-image-data/` whose photo does not show its query, such as a night sky saved for "sand".

Each web photo (Wikimedia, Openverse) is scored with [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) against `a photo of {query}`. A pair is kept when both hold:

- **`--min-score`** (default 0.22): similarity between the photo and its own query.
- **`--top-k`** (default 50): its own query is among the 50 best-matching of all 462 queries in the manifest.

Open Images pairs are never dropped; their queries come from human-drawn boxes.

## Architecture

### What is taken from each pair

| Manifest column | Used for |
|---|---|
| `input_path` | The 256×256 photo given to CLIP |
| `query` | The category the photo was downloaded for, e.g. "owl" |
| `provider` | Open Images rows are skipped |
| `id` | Key in `clip_scores.csv` |

### The 462 queries

The photo is compared with every distinct `query` in the manifest, not only its own:

| Source | Queries | Examples |
|---|---|---|
| Web queries | 253 | "magnifying glass", "waterfall", "arcade cabinet" |
| Open Images | 304 | "dog", "man riding horse", "running man" |
| In both lists | −95 | "apple", "bicycle", "boat" |
| **Total** | **462** | |

Each query becomes the text `a photo of {query}`. The [CLIP paper](https://arxiv.org/abs/2103.00020) (section 3.1.4) found this kind of prompt matches photos better than the bare word.

### What CLIP outputs

CLIP has two encoders, trained together on 400 million image–text pairs so that a photo and a text that match get similar outputs:

- **Image encoder:** photo → 512 numbers
- **Text encoder:** text → 512 numbers

Both outputs are scaled to length 1. The dot product of a photo's 512 numbers and a text's 512 numbers is then their similarity (cosine similarity): higher means a better match. On this data it ranges from about 0.1 to 0.35, so I chose 0.22 as a baseline.

### How score and rank are calculated

```
 photo (input_path)                      all 462 queries
        |                                "owl", "helmet", "vase", ...
        v                                       |
 CLIP image encoder                             v
        |                          "a photo of owl", "a photo of helmet", ...
        v                                       |
   512 numbers                                  v
        |                               CLIP text encoder
        |                                       |
        |                                       v
        |                              462 x 512 numbers
        |                                       |
        +----------> dot product <--------------+
                          |
                          v
      462 similarities, one per query (higher = better match)
      helmet 0.270, vase 0.269, ..., owl 0.229, ...
                          |
                          v          the row's own query is "owl"
      score = 0.229  (similarity to "owl")
      rank  = 89     (queries with a higher similarity than "owl")
                          |
                          v
      keep if score >= 0.22 and rank < 50   ->   dropped
```

- **score**: the similarity between the photo and its own query.
- **rank**: how many of the 462 queries have a higher similarity than the score. 0 means its own query is the best match. `rank < 50` means its own query is in the photo's top 50.

The text side is calculated once. Photos are processed 256 at a time, giving a 256 × 462 table of similarities per batch.

### Examples from the dataset

| Saved for | Score | Rank | Best-matching queries | Result |
|---|---|---|---|---|
| sailing ship | 0.302 | 0 | sailing ship 0.302, pirate 0.271, anchor 0.267 | kept |
| owl (a bronze artefact) | 0.229 | 89 | helmet 0.270, vase 0.269, hammer 0.267 | dropped by rank |
| duck (a breakfast plate) | 0.208 | 5 | egg 0.251, diner 0.222, rabbit 0.215 | dropped by score |

### Why two checks

- **Score** asks whether the photo matches its own query well enough. The owl photo passes it, but 89 queries match it better.
- **Rank** compares queries on the same photo. The duck photo ranks 5th because no query matches a plate of food strongly, but its score is low.
- Rank is unfair to general queries. A correct "person" photo also matches "man", "woman" and "boy", and each pushes "person" one place down.

### How the defaults were chosen

Open Images queries are known to be right, so their scores show what a correct pair looks like:

- 5% of Open Images pairs score below 0.221.
- 96% have a rank below 50.
- 94% pass both checks, against 80% of web pairs.

## Run

This is step 3 of [`src/setup_data.py`](../../src/setup_data.py). To run it alone, for example to try other cut-offs, from the project root:

```bash
uv run python -m scripts.refine.clip_filter
uv run python -m scripts.refine.clip_filter --min-score 0.24 --top-k 20   # stricter
```

| Run | What happens | Time |
|---|---|---|
| First | Scores all ~73k web photos with CLIP | ~5 minutes on an Apple-silicon GPU |
| Later | Reads saved scores; scores only pairs added since | Seconds |

### Files

| File | Contents |
|---|---|
| `pixel-image-data/clean.csv` | The manifest without dropped pairs. Rewritten on every run. |
| `pixel-image-data/clip_scores.csv` | One row per scored web pair: `id`, `query`, `score`, `rank`, `query_list` |

`manifest.csv` and the image files are never changed.

`query_list` names the query list the rank was computed against, as `<count>-<hash>`, e.g. `462-45672d12`. If the manifest's queries change later, the script prints a warning, because those ranks are now out of date:

```text
warning: 73275 ranks in pixel-image-data/clip_scores.csv were computed against 462 queries (462-45672d12), but the manifest now has 461 queries (461-164c2044).
Ranks depend on the query list; delete pixel-image-data/clip_scores.csv and rerun to rescore.
```
