# Refine

`clip_filter.py` drops photo pairs in `pixel-image-data/` that should not be trained on or published, and writes the rest to `clean.csv`. Each pair is dropped for the first reason that applies:

| Reason | Check | Applies to | Dropped on current data |
|---|---|---|---|
| `nudity` | The photo matches nudity prompts better than neutral prompts by at least `--nudity` (default 0.02) | All photos | 703 |
| `graphic` | The same with prompts for gore, dead bodies and injuries, `--graphic` (default 0.0395) | All photos | 101 |
| `mismatch` | Similarity to its own query below `--min-score` (default 0.22), or rank of its own query `--top-k` (default 50) or worse | Web photos (Wikimedia, Openverse) | 14,121 |
| `no author` | CC BY licence but no author to credit | All photos | 204 |
| `email author` | The author field is an email address | All photos | 61 |

On the current data, 84,290 of 99,480 pairs are kept. Open Images pairs skip the mismatch check because their queries come from human-drawn boxes.

The checks use [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32).

## Architecture

### What is taken from each pair

| Manifest column | Used for |
|---|---|
| `input_path` | The 256×256 photo given to CLIP |
| `query` | The category the photo was downloaded for, e.g. "owl" |
| `provider` | Open Images rows skip the mismatch check |
| `license`, `author` | The attribution checks |
| `id` | Key in `clip_embeddings.pt` |

The `title` column is not used. Titles are free text from the uploader, such as `File:Butterfly 23 (8254394207).jpg`, and can say anything.

### What CLIP outputs

CLIP has two encoders, trained together on 400 million image–text pairs so that a photo and a text that match get similar outputs:

- **Image encoder:** photo → 512 numbers
- **Text encoder:** text → 512 numbers

Both outputs are scaled to length 1. The dot product of a photo's 512 numbers and a text's 512 numbers is then their similarity (cosine similarity): higher means a better match. On this data it ranges from about 0.1 to 0.35, so I chose 0.22 as a baseline.

Every photo is embedded once and cached in `clip_embeddings.pt`. Each run embeds only the texts and multiplies them with the cached photos, so all scores are recalculated against the current queries and prompts.

### Mismatch check

#### The 462 queries

The photo is compared with every distinct `query` in the manifest, not only its own:

| Source | Queries | Examples |
|---|---|---|
| Web queries | 253 | "magnifying glass", "waterfall", "arcade cabinet" |
| Open Images | 304 | "dog", "man riding horse", "running man" |
| In both lists | −95 | "apple", "bicycle", "boat" |
| **Total** | **462** | |

Each query becomes the text `a photo of {query}`. The [CLIP paper](https://arxiv.org/abs/2103.00020) (section 3.1.4) found this kind of prompt matches photos better than the bare word.

#### How score and rank are calculated

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

#### Examples from the dataset

| Saved for | Score | Rank | Best-matching queries | Result |
|---|---|---|---|---|
| sailing ship | 0.302 | 0 | sailing ship 0.302, pirate 0.271, anchor 0.267 | kept |
| owl (a bronze artefact) | 0.229 | 89 | helmet 0.270, vase 0.269, hammer 0.267 | dropped by rank |
| duck (a breakfast plate) | 0.208 | 5 | egg 0.251, diner 0.222, rabbit 0.215 | dropped by score |

#### Why two checks

- **Score** asks whether the photo matches its own query well enough. The owl photo passes it, but 89 queries match it better.
- **Rank** compares queries on the same photo. The duck photo ranks 5th because no query matches a plate of food strongly, but its score is low.
- Rank is unfair to general queries. A correct "person" photo also matches "man", "woman" and "boy", and each pushes "person" one place down.

#### How the defaults were chosen

Open Images queries are known to be right, so their scores show what a correct pair looks like:

- 5% of Open Images pairs score below 0.221.
- 96% have a rank below 50.
- 94% pass both checks, against 80% of web pairs.

### Content check (nudity, graphic)

Each photo is compared with three groups of prompts:

| Group | Prompts |
|---|---|
| nudity | "a photo of a naked person", "a nude body", "a pornographic photo" |
| graphic | "a photo of gore and blood", "a photo of a dead body", "a graphic injury" |
| neutral | "a photo", "a photo of an object", "a photo of a person" |

The **margin** is the best similarity in a group minus the best neutral similarity. Every photo is somewhat similar to any "a photo of …" text, so the margin measures how much more the photo looks like the flagged content than like an ordinary photo.

```
 photo saved for "lizard"
        |
        +-- graphic prompts:  gore 0.200, dead body 0.263, injury 0.194  ->  best 0.263
        +-- neutral prompts:  photo 0.227, object 0.227, person 0.223    ->  best 0.227
        |
        v
 margin = 0.263 - 0.227 = 0.036   >= 0.03   ->   dropped (graphic)
```

#### How the defaults were chosen

I looked at random samples of 48 photos in each margin band:

| Band | What the samples showed |
|---|---|
| nudity ≥ 0.03 | About half nude: mostly art (statues, paintings, engravings), also real people (nude beaches, naked bike rides) and one explicit photo |
| nudity 0.02–0.03 | Mostly harmless; a few nude or in underwear |
| nudity 0.015–0.02 | Almost all harmless; some underwear |
| graphic ≥ 0.035 | About half graphic: hunted animals, carcasses, zombie make-up, one real casualty scene |
| graphic 0.025–0.035 | Mostly harmless; a few dead animals |

Then the dropped photos were reviewed on the `--sheets` contact sheets:

- **nudity 0.02** was kept. Photos between 0.02 and 0.035 still include nude and topless adults, erotic art and children in a bath.
- **graphic** was raised from 0.03 to 0.0395, the score of the 100th highest photo. Below it, flagged photos were mostly harmless (politicians, sports) or mild (animal skeletons, carcasses, zombie make-up).

The defaults drop about 0.8% of photos, including many harmless nude-flagged ones, so that few unsuitable photos are left. The check is automatic and can still miss some.

### Attribution checks

- **no author:** CC BY requires crediting the creator. Without an author, a pair cannot be credited, so it is dropped.
- **email author:** some Flickr and Wikimedia authors use an email address as their name. These pairs are dropped so the dataset does not republish email addresses.

## Run

This is step 3 of [`src/setup_data.py`](../../src/setup_data.py). To run it alone, for example to try other cut-offs, from the project root:

```bash
uv run python -m scripts.refine.clip_filter
uv run python -m scripts.refine.clip_filter --sheets                      # also save review sheets
uv run python -m scripts.refine.clip_filter --min-score 0.24 --top-k 20   # stricter mismatch check
uv run python -m scripts.refine.clip_filter --nudity 0.015                # stricter nudity check
```

| Run | What happens | Time |
|---|---|---|
| First | Embeds all ~99k photos with CLIP | ~12 minutes on an Apple-silicon GPU |
| Later | Embeds only photos added since; recalculates every score | ~6 seconds |

### Files

| File | Contents |
|---|---|
| `pixel-image-data/clean.csv` | The manifest without dropped pairs. Rewritten on every run. [`scripts/hub`](../hub/README.md) uploads these rows. |
| `pixel-image-data/dropped.csv` | One row per dropped pair: `id`, `provider`, `query`, `reason`, `query_score`, `rank`, `nudity`, `graphic` |
| `pixel-image-data/clip_embeddings.pt` | Cached CLIP photo vectors: `ids` and a matching N × 512 float16 tensor (~100 MB) |
| `pixel-image-data/review/` | With `--sheets`: `nudity-1.png`, `graphic-1.png`, … with 100 dropped photos each, highest margin first, labelled with margin, query and id |

`manifest.csv` and the image files are never changed.

Delete `clip_embeddings.pt` only after switching to another CLIP model or changing how photos are prepared. A changed photo gets a new `id` (ids are hashes of the downloaded image), so it is embedded again anyway. The file is loaded with `weights_only=True`, which cannot run code.
