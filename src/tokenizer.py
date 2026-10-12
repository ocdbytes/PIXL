"""Caption tokenizer: byte-level byte-pair encoding (BPE), trained on pixl's captions.

Text is lowercased and split into words. Each word starts as its UTF-8 bytes with
a leading space, so tokens know where words begin. Training repeatedly merges the
most frequent adjacent pair of tokens into a new token until the vocabulary is
full (Sennrich et al., 2016, https://arxiv.org/abs/1508.07909). Common words end
up as one token; a word never seen in training splits into smaller pieces, down
to single bytes, so any text can be encoded.

Token ids: 0 <pad>, 1 <bos>, 2 <eos>, 3-258 the 256 bytes, then one id per merge.
Saved to src/tokenizer.json; a trained model only works with the tokenizer it was
trained with, so retrain it only before training the model from scratch.

    uv run python -m src.tokenizer
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from datasets import load_dataset

from .data import COLUMNS
from .preprocess import REPO

TOKENIZER = Path("src/tokenizer.json")
PAD, BOS, EOS = 0, 1, 2
SPECIALS = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS}
# Id of byte 0; byte b is FIRST_BYTE + b, and merges follow the 256 bytes.
FIRST_BYTE = len(SPECIALS)
FIRST_MERGE = FIRST_BYTE + 256

Pair = tuple[int, int]


@dataclass
class Args(argparse.Namespace):
    vocab_size: int = 1024


def words(text: str) -> list[bytes]:
    """Splits text into lowercased words, each with a leading space."""
    return [f" {word}".encode() for word in text.lower().split()]


def merge(ids: list[int], pair: Pair, new: int) -> list[int]:
    """Replaces every occurrence of pair in ids with new, left to right."""
    merged: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            merged.append(new)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


def train(texts: Iterable[str], vocab_size: int) -> list[Pair]:
    """Learns BPE merges from texts.

    Args:
        texts: Training captions, repeated as often as they occur in training.
        vocab_size: Size of the finished vocabulary, special tokens included.

    Returns:
        Merges in the order learned; merge i creates token FIRST_MERGE + i.
    """
    counts = Counter(word for text in texts for word in words(text))
    pieces = {word: [FIRST_BYTE + byte for byte in word] for word in counts}
    merges: list[Pair] = []
    while FIRST_MERGE + len(merges) < vocab_size:
        pairs: Counter[Pair] = Counter()
        for word, ids in pieces.items():
            for pair in zip(ids, ids[1:]):
                pairs[pair] += counts[word]
        if not pairs:
            break  # every word is already one token
        # Most frequent pair; ties go to the smallest ids, so training is repeatable.
        best = min(pairs, key=lambda pair: (-pairs[pair], pair))
        merges.append(best)
        new = FIRST_MERGE + len(merges) - 1
        pieces = {word: merge(ids, best, new) for word, ids in pieces.items()}
    return merges


class Tokenizer:
    """Encodes captions as token ids and back, using learned merges."""

    def __init__(self, merges: list[Pair]) -> None:
        self.merges: list[Pair] = merges
        # A pair's id is also its rank: earlier merges are applied first.
        self.ids: dict[Pair, int] = {
            pair: FIRST_MERGE + i for i, pair in enumerate(merges)
        }
        self.pieces: dict[int, bytes] = {
            FIRST_BYTE + byte: bytes([byte]) for byte in range(256)
        }
        for pair, new in self.ids.items():
            self.pieces[new] = self.pieces[pair[0]] + self.pieces[pair[1]]
        self.word = lru_cache(maxsize=None)(self._word)

    @property
    def vocab_size(self) -> int:
        return FIRST_MERGE + len(self.merges)

    def _word(self, word: bytes) -> tuple[int, ...]:
        """Encodes one word by applying merges in the order they were learned."""
        ids = [FIRST_BYTE + byte for byte in word]
        while len(ids) > 1:
            pair = min(
                zip(ids, ids[1:]), key=lambda p: self.ids.get(p, self.vocab_size)
            )
            if pair not in self.ids:
                break
            ids = merge(ids, pair, self.ids[pair])
        return tuple(ids)

    def encode(self, text: str) -> list[int]:
        """Turns text into token ids, without <bos> or <eos>."""
        return [i for word in words(text) for i in self.word(word)]

    def decode(self, ids: Iterable[int]) -> str:
        """Turns token ids back into text; special tokens are dropped."""
        data = b"".join(self.pieces[i] for i in ids if i >= FIRST_BYTE)
        return data.decode("utf-8", errors="replace").strip()

    def piece(self, token: int) -> str:
        """Shows a token as text, e.g. " grass", for reading the vocabulary."""
        if token < FIRST_BYTE:
            return next(name for name, i in SPECIALS.items() if i == token)
        return self.pieces[token].decode("utf-8", errors="replace")

    def save(self, path: Path = TOKENIZER) -> None:
        data = {
            "specials": SPECIALS,
            "merges": self.merges,
            # For reading only; load() rebuilds the pieces from the merges.
            "tokens": [self.piece(i) for i in range(self.vocab_size)],
        }
        _ = path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = TOKENIZER) -> Tokenizer:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls([(first, second) for first, second in data["merges"]])


def captions(split: str) -> dict[str, list[str]]:
    """Reads every caption of one split from the Hugging Face dataset, per config."""
    return {
        config: load_dataset(REPO, config, split=split)[column]
        for config, (column, _) in COLUMNS.items()
    }


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _ = parser.add_argument("--vocab-size", type=int, default=Args.vocab_size)
    return parser.parse_args(namespace=Args())


def main() -> None:
    args = parse_args()
    train_captions = captions("train")
    tokenizer = Tokenizer(
        train(
            [text for texts in train_captions.values() for text in texts],
            args.vocab_size,
        )
    )
    tokenizer.save()
    print(f"saved {tokenizer.vocab_size} tokens to {TOKENIZER}")

    # Checks: every caption decodes back to itself; lengths per config.
    for split, by_config in (("train", train_captions), ("test", captions("test"))):
        for config, texts in by_config.items():
            lengths = [len(tokenizer.encode(text)) for text in texts]
            exact = sum(
                tokenizer.decode(tokenizer.encode(text))
                == " ".join(text.lower().split())
                for text in texts
            )
            print(
                f"{config}/{split}: round trip {exact}/{len(texts)},"
                f" tokens per caption mean {sum(lengths) / len(lengths):.1f}, max {max(lengths)}"
            )
    for text in (
        "a green pine tree left of a wooden barrel on grass",
        "a purple dragon above a castle",
        "basketball on grass",
    ):
        print(f"{text!r} -> {[tokenizer.piece(i) for i in tokenizer.encode(text)]}")


if __name__ == "__main__":
    main()
