"""
Knowledge Graph dataset loaders for FB15k-237 and WN18RR.

Supports automatic download from multiple mirrors.
"""

import os
from typing import Tuple, Dict, List

import torch

# ── Dataset registry ────────────────────────────────────

DATASET_REGISTRY: Dict[str, dict] = {
    "fb15k237": {
        "name": "FB15k-237",
        "files": {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"},
        "mirrors": [
            "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237",
        ],
    },
    "wn18rr": {
        "name": "WN18RR",
        "files": {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"},
        "mirrors": [
            "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/wn18rr",
        ],
    },
}


# ── Dataset container ───────────────────────────────────

class KGDataset:
    """
    Knowledge Graph dataset container.

    Attributes:
        name: Dataset name.
        entity2id: Dict mapping entity strings to integer IDs.
        relation2id: Dict mapping relation strings to integer IDs.
        num_entities: Total unique entities.
        num_relations: Total unique relation types.
        train_triples: (N_train, 3) LongTensor.
        valid_triples: (N_valid, 3) LongTensor.
        test_triples:  (N_test, 3) LongTensor.
    """

    def __init__(
        self,
        data_dir: str,
        entity2id: Dict[str, int],
        relation2id: Dict[str, int],
        train_triples: torch.LongTensor,
        valid_triples: torch.LongTensor,
        test_triples: torch.LongTensor,
        name: str = "",
    ):
        self.data_dir = data_dir
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.id2entity = {v: k for k, v in entity2id.items()}
        self.id2relation = {v: k for k, v in relation2id.items()}
        self.num_entities = len(entity2id)
        self.num_relations = len(relation2id)
        self.train_triples = train_triples
        self.valid_triples = valid_triples
        self.test_triples = test_triples
        self.name = name

    def __repr__(self) -> str:
        return (
            f"KGDataset({self.name}, entities={self.num_entities}, "
            f"relations={self.num_relations}, "
            f"train={self.train_triples.shape[0]}, "
            f"valid={self.valid_triples.shape[0]}, "
            f"test={self.test_triples.shape[0]})"
        )


# ── Download helpers ────────────────────────────────────

def _download_file(url: str, dest: str) -> None:
    import urllib.request
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)


def _download_with_fallback(
    filename: str, dest_dir: str, mirrors: List[str], force: bool = False,
) -> str:
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest) and not force:
        return dest
    for mirror in mirrors:
        url = f"{mirror}/{filename}"
        try:
            _download_file(url, dest)
            return dest
        except Exception as e:
            print(f"    Mirror failed: {e}")
            continue
    raise RuntimeError(
        f"Failed to download {filename}. Place files manually in {dest_dir}/"
    )


# ── Parsing ─────────────────────────────────────────────

def _parse_triples_file(
    file_path: str, entity2id: Dict[str, int], relation2id: Dict[str, int],
) -> torch.LongTensor:
    triples = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            triples.append([entity2id[parts[0]], relation2id[parts[1]], entity2id[parts[2]]])
    return torch.LongTensor(triples)


def _build_vocab(
    train_file: str, valid_file: str, test_file: str,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    entities, relations = set(), set()
    for fp in [train_file, valid_file, test_file]:
        with open(fp, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                entities.add(parts[0])
                entities.add(parts[2])
                relations.add(parts[1])
    entity2id = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}
    return entity2id, relation2id


# ── Public API ──────────────────────────────────────────

def load_dataset(
    dataset_key: str, data_dir: str, force_download: bool = False,
) -> KGDataset:
    """Load a KG dataset by key ("fb15k237" or "wn18rr")."""
    info = DATASET_REGISTRY.get(dataset_key)
    if info is None:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. "
            f"Choose from: {list(DATASET_REGISTRY.keys())}"
        )

    dest_dir = os.path.join(data_dir, dataset_key)
    os.makedirs(dest_dir, exist_ok=True)

    for fname in info["files"].values():
        _download_with_fallback(fname, dest_dir, info["mirrors"], force=force_download)

    paths = {s: os.path.join(dest_dir, f) for s, f in info["files"].items()}
    entity2id, relation2id = _build_vocab(paths["train"], paths["valid"], paths["test"])

    train = _parse_triples_file(paths["train"], entity2id, relation2id)
    valid = _parse_triples_file(paths["valid"], entity2id, relation2id)
    test = _parse_triples_file(paths["test"], entity2id, relation2id)

    dataset = KGDataset(dest_dir, entity2id, relation2id, train, valid, test, name=info["name"])
    print(f"Loaded {info['name']}: {dataset}")
    return dataset


def load_fb15k237(data_dir: str, force_download: bool = False) -> KGDataset:
    """Load FB15k-237."""
    return load_dataset("fb15k237", data_dir, force_download)


def load_wn18rr(data_dir: str, force_download: bool = False) -> KGDataset:
    """Load WN18RR."""
    return load_dataset("wn18rr", data_dir, force_download)
