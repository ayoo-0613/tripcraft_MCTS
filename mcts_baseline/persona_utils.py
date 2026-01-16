from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.metrics.pairwise import cosine_similarity


_BERT_TOKENIZER = None
_BERT_MODEL = None
_BERT_TRIED = False

_PERSONA_CACHE: Dict[str, Dict[str, Any]] = {}
_POI_CACHE: Dict[str, Any] = {}


def _get_bert():
    global _BERT_TOKENIZER, _BERT_MODEL, _BERT_TRIED
    if _BERT_TRIED:
        return _BERT_TOKENIZER, _BERT_MODEL
    _BERT_TRIED = True
    try:
        from transformers import BertTokenizer, BertModel

        try:
            _BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
            _BERT_MODEL = BertModel.from_pretrained("bert-base-uncased", local_files_only=True)
        except TypeError:
            _BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
            _BERT_MODEL = BertModel.from_pretrained("bert-base-uncased")
        if _BERT_MODEL is not None:
            _BERT_MODEL.eval()
    except Exception:
        _BERT_TOKENIZER = None
        _BERT_MODEL = None
    return _BERT_TOKENIZER, _BERT_MODEL


def _get_bert_embedding(text: str, tokenizer: Any, model: Any):
    if tokenizer is None or model is None:
        return None
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()


def _extract_persona_components(persona: str) -> Dict[str, str]:
    components = {
        "Traveler Type": None,
        "Purpose of Travel": None,
        "Spending Preference": None,
        "Location Preference": None,
    }
    for key in components.keys():
        start_idx = persona.find(key + ":") + len(key) + 1
        end_idx = persona.find(";", start_idx)
        if end_idx == -1:
            end_idx = len(persona)
        components[key] = persona[start_idx:end_idx].strip()
    return components


def extract_poi_name(poi: str) -> str:
    if "stay" in poi:
        return poi.split("stay")[0].strip()[:-1]
    return poi.split("visit")[0].strip()[:-1]


def candidate_poi_name(name: str, kind: str) -> str:
    if not name:
        return ""
    return extract_poi_name(f"{name}, {kind}")


def get_persona_embeddings(persona: str) -> Optional[Dict[str, Any]]:
    if persona in _PERSONA_CACHE:
        return _PERSONA_CACHE[persona]
    tokenizer, model = _get_bert()
    if tokenizer is None or model is None:
        return None
    components = _extract_persona_components(persona or "")
    embeddings: Dict[str, Any] = {}
    for key, value in components.items():
        emb = _get_bert_embedding(value, tokenizer, model)
        if emb is None:
            return None
        embeddings[key] = emb
    _PERSONA_CACHE[persona] = embeddings
    return embeddings


def get_poi_embedding(poi_name: str) -> Optional[Any]:
    if poi_name in _POI_CACHE:
        return _POI_CACHE[poi_name]
    tokenizer, model = _get_bert()
    if tokenizer is None or model is None:
        return None
    emb = _get_bert_embedding(poi_name, tokenizer, model)
    if emb is None:
        return None
    _POI_CACHE[poi_name] = emb
    return emb


def average_persona_similarity(poi_emb: Any, persona_embeddings: Dict[str, Any]) -> float:
    total = 0.0
    for persona_emb in persona_embeddings.values():
        total += cosine_similarity([persona_emb], [poi_emb])[0][0]
    return total / float(len(persona_embeddings)) if persona_embeddings else 0.0

