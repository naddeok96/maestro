"""Realistic NER datasets backed by CoNLL-style corpora."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from datasets import DatasetDict


DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "MAESTRO_DATA_ROOT",
        Path(__file__).resolve().parents[2] / "data",
    )
)
HF_CACHE = Path(
    os.environ.get(
        "MAESTRO_HF_CACHE",
        DEFAULT_DATA_ROOT / "hf_cache",
    )
)
HF_CACHE.mkdir(parents=True, exist_ok=True)


__all__ = ["NERConfig", "build_ner_dataset"]

_MISSING_DATASETS_MSG = (
    "The 'datasets' package is required for the NER builder. "
    "Please install it via `pip install datasets` or run "
    "`scripts/download_datasets.sh`."
)


def _load_hf_loader():
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - handled via requirements
        raise ImportError(_MISSING_DATASETS_MSG) from exc
    return load_dataset


@dataclass
class NERConfig:
    dataset_kind: str
    train_size: int
    val_size: int
    probe_size: int
    noise: float
    max_sequence_length: int = 32
    entity_injection_prob: float = 0.0
    focus_entities: Sequence[str] | None = None
    lowercase: bool = True


class TokenSequenceDataset(Dataset):
    """Simple tensor-backed dataset for token/tag sequences."""

    def __init__(self, tokens: np.ndarray, tags: np.ndarray) -> None:
        self.tokens = torch.from_numpy(tokens).long()
        self.tags = torch.from_numpy(tags).long()

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, index: int):
        return self.tokens[index], self.tags[index]


_VOCAB_CACHE: Dict[Tuple[str, bool], Dict[str, int]] = {}


def _build_vocab(
    corpus: DatasetDict, lowercase: bool, *, min_freq: int = 1
) -> Dict[str, int]:
    cache_key = (corpus["train"].info.builder_name, lowercase)
    if cache_key in _VOCAB_CACHE:
        return _VOCAB_CACHE[cache_key]
    counter: Dict[str, int] = {}
    for split in corpus.values():
        tokens_list = split["tokens"]
        for tokens in tokens_list:
            for token in tokens:
                key = token.lower() if lowercase else token
                counter[key] = counter.get(key, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if freq < min_freq:
            continue
        if token not in vocab:
            vocab[token] = len(vocab)
    _VOCAB_CACHE[cache_key] = vocab
    return vocab


def _encode_example(
    tokens: Sequence[str],
    tags: Sequence[int],
    vocab: Dict[str, int],
    *,
    lowercase: bool,
    max_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    token_ids = np.full(max_length, vocab["<pad>"], dtype=np.int64)
    tag_ids = np.zeros(max_length, dtype=np.int64)
    limit = min(len(tokens), max_length)
    for idx in range(limit):
        token = tokens[idx].lower() if lowercase else tokens[idx]
        token_ids[idx] = vocab.get(token, vocab["<unk>"])
        tag_ids[idx] = int(tags[idx])
    return token_ids, tag_ids


def _inject_entities(
    tag_ids: np.ndarray,
    rng: np.random.Generator,
    num_tags: int,
) -> None:
    if tag_ids.size == 0:
        return
    span = max(1, int(tag_ids.size * 0.2))
    start = int(rng.integers(0, max(1, tag_ids.size - span + 1)))
    end = min(tag_ids.size, start + span)
    new_tag = int(rng.integers(1, num_tags))
    tag_ids[start:end] = new_tag


def _apply_sequence_noise(
    tag_ids: np.ndarray, noise_rate: float, rng: np.random.Generator, num_tags: int
) -> np.ndarray:
    if noise_rate <= 0.0:
        return tag_ids
    noisy = tag_ids.copy()
    mask = rng.random(size=noisy.shape) < noise_rate
    if mask.any():
        noisy[mask] = rng.integers(0, num_tags, size=int(mask.sum()))
    return noisy


def _collect_examples(
    split,
    focus_ids: Sequence[int] | None,
) -> List[Dict[str, Sequence]]:
    examples: List[Dict[str, Sequence]] = []
    for tokens, tags in zip(split["tokens"], split["ner_tags"]):
        if focus_ids:
            if not any(tag in focus_ids for tag in tags):
                continue
        examples.append({"tokens": tokens, "ner_tags": tags})
    return examples if examples else [
        {"tokens": tokens, "ner_tags": tags}
        for tokens, tags in zip(split["tokens"], split["ner_tags"])
    ]


def _sample_examples(
    examples: Sequence[Dict[str, Sequence]],
    size: int,
    rng: np.random.Generator,
) -> List[Dict[str, Sequence]]:
    if not examples:
        raise ValueError("No examples available to sample.")
    replace = size > len(examples)
    indices = rng.choice(len(examples), size=size, replace=replace)
    return [examples[int(i)] for i in indices]


def _resolve_focus_ids(
    focus_entities: Sequence[str] | None,
    tag_names: Sequence[str],
) -> List[int] | None:
    if not focus_entities:
        return None
    name_to_id = {name: idx for idx, name in enumerate(tag_names)}
    ids = [name_to_id[name] for name in focus_entities if name in name_to_id]
    return ids or None


def _encode_split(
    examples: Sequence[Dict[str, Sequence]],
    size: int,
    rng: np.random.Generator,
    *,
    vocab: Dict[str, int],
    lowercase: bool,
    max_length: int,
    num_tags: int,
    noise: float,
    entity_injection_prob: float,
) -> Tuple[np.ndarray, np.ndarray]:
    sampled = _sample_examples(examples, size, rng)
    tokens = np.zeros((len(sampled), max_length), dtype=np.int64)
    tags = np.zeros((len(sampled), max_length), dtype=np.int64)
    for idx, example in enumerate(sampled):
        token_ids, tag_ids = _encode_example(
            example["tokens"],
            example["ner_tags"],
            vocab,
            lowercase=lowercase,
            max_length=max_length,
        )
        if entity_injection_prob > 0.0 and rng.random() < entity_injection_prob:
            _inject_entities(tag_ids, rng, num_tags)
        if noise > 0.0:
            tag_ids = _apply_sequence_noise(tag_ids, noise, rng, num_tags)
        tokens[idx] = token_ids
        tags[idx] = tag_ids
    return tokens, tags


def build_ner_dataset(
    name: str,
    config: NERConfig,
    seed: int,
) -> Dict[str, Dataset]:
    """Materialise NER datasets from a HuggingFace corpus."""

    load_dataset = _load_hf_loader()
    corpus = load_dataset(config.dataset_kind, cache_dir=str(HF_CACHE))
    tag_feature = corpus["train"].features["ner_tags"].feature
    tag_names = list(tag_feature.names)
    num_tags = len(tag_names)
    vocab = _build_vocab(corpus, lowercase=config.lowercase)
    rng = np.random.default_rng(seed)
    focus_ids = _resolve_focus_ids(config.focus_entities, tag_names)
    train_examples = _collect_examples(corpus["train"], focus_ids)
    val_examples = _collect_examples(corpus["validation"], focus_ids)
    test_examples = _collect_examples(corpus["test"], focus_ids)
    train_tokens, train_tags = _encode_split(
        train_examples,
        config.train_size,
        rng,
        vocab=vocab,
        lowercase=config.lowercase,
        max_length=config.max_sequence_length,
        num_tags=num_tags,
        noise=config.noise,
        entity_injection_prob=config.entity_injection_prob,
    )
    val_tokens, val_tags = _encode_split(
        val_examples,
        config.val_size,
        rng,
        vocab=vocab,
        lowercase=config.lowercase,
        max_length=config.max_sequence_length,
        num_tags=num_tags,
        noise=0.0,
        entity_injection_prob=0.0,
    )
    probe_tokens, probe_tags = _encode_split(
        test_examples,
        config.probe_size,
        rng,
        vocab=vocab,
        lowercase=config.lowercase,
        max_length=config.max_sequence_length,
        num_tags=num_tags,
        noise=0.0,
        entity_injection_prob=0.0,
    )
    metadata = {
        "num_tags": num_tags,
        "vocab_size": len(vocab),
        "sequence_length": config.max_sequence_length,
        "input_shape": (config.max_sequence_length,),
        "input_type": "text",
        "dataset_kind": config.dataset_kind,
        "focus_entities": list(config.focus_entities) if config.focus_entities else None,
        "name": name,
    }
    return {
        "train": TokenSequenceDataset(train_tokens, train_tags),
        "val": TokenSequenceDataset(val_tokens, val_tags),
        "probe": TokenSequenceDataset(probe_tokens, probe_tags),
        "metadata": metadata,
    }
