# src/features.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
import joblib

RANDOM_STATE = 42  # global seed for reproducibility

# -----------------------------
# Lexicons (same as notebook)
# -----------------------------
PROFANITY  = {"fuck","fucking","shit","bitch","cunt","ass","dick","suck","sucks","moron","stupid"}
VIOLENCE   = {"kill","die","murder","hurt","destroy","stab","shoot"}
PRON_2ND   = {"you","your","u","ur","you're","youre"}
INTENS     = {"very","really","so","extremely","totally","absolutely","super"}
POLITE     = {"please","thank","thanks","appreciate"}
NEGATE     = {
    "not","no","never",
    "don't","dont","can't","cant","won't","wont","isn't","isnt","didn't","didnt",
    "doesn't","doesnt","shouldn't","shouldnt","couldn't","couldnt","wouldn't","wouldnt",
    "ain't","aint","n't"
}
IDENTITY   = {"jew","jewish","muslim","christian","black","white","gay","lesbian","mexican","mexicans"}
GENERALIZ  = {"these","those","all","every","always","everyone","nobody"}
HEDGES_UNI = {"maybe","perhaps","seems","apparently","probably","possibly","likely"}
HEDGE_PHRASES = {"i think","i guess","sort of","kind of"}

URL_RE     = re.compile(r"(https?://\S|www\.\S)", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")

# -----------------------------
# Cue feature builders
# -----------------------------
def _tokens(series: pd.Series) -> List[List[str]]:
    """Lowercase, safe-cast to str, split on whitespace."""
    return series.fillna("").astype(str).str.lower().str.split().tolist()

def _rate_per_k(tokens: List[str], lexicon: set, per: int = 1000, min_tokens: int = 20) -> float:
    """Per-1k rate with length smoothing to avoid spikes on very short texts."""
    n = max(len(tokens), min_tokens)
    return 0.0 if n == 0 else per * sum(t in lexicon for t in tokens) / n

def _has_any(tokens: List[str], lexicon: set) -> int:
    return int(any(t in lexicon for t in tokens))

def _has_any_bigram(tokens: List[str], phrases: set) -> int:
    s = " ".join(tokens)
    return int(any(p in s for p in phrases))

def _raw_counts(text: str) -> Tuple[int,int,int,int,int]:
    s = "" if not isinstance(text, str) else text
    had_url     = int(bool(URL_RE.search(s)))
    had_mention = int(bool(MENTION_RE.search(s)))
    hashtag_cnt = len(HASHTAG_RE.findall(s))
    bangs       = s.count("!")
    qmarks      = s.count("?")
    return had_url, had_mention, hashtag_cnt, bangs, qmarks

CUE_COLS = [
    "rate_profane","rate_violence","rate_2nd","rate_intens","rate_negate","rate_polite",
    "rate_identity","rate_general","rate_hedge",
    "has_profane","has_identity","has_hedge_bigram",
    "n_tokens","avg_tok_len","had_url","had_mention","hashtag_count","bangs","qmarks"
]

def build_psych_features_smooth(
    df: pd.DataFrame, tokens_col: str, raw_col: str,
    per: int = 1000, min_tokens: int = 20
) -> pd.DataFrame:
    """
    Build interpretable cue features with per-1k smoothing and simple binary flags.
    Returns a dense DataFrame with fixed column order CUE_COLS.
    """
    toks_all = _tokens(df[tokens_col])
    rows = []
    for i, ts in enumerate(toks_all):
        rows.append({
            "rate_profane":  _rate_per_k(ts, PROFANITY,  per, min_tokens),
            "rate_violence": _rate_per_k(ts, VIOLENCE,   per, min_tokens),
            "rate_2nd":      _rate_per_k(ts, PRON_2ND,   per, min_tokens),
            "rate_intens":   _rate_per_k(ts, INTENS,     per, min_tokens),
            "rate_negate":   _rate_per_k(ts, NEGATE,     per, min_tokens),
            "rate_polite":   _rate_per_k(ts, POLITE,     per, min_tokens),
            "rate_identity": _rate_per_k(ts, IDENTITY,   per, min_tokens),
            "rate_general":  _rate_per_k(ts, GENERALIZ,  per, min_tokens),
            "rate_hedge":    _rate_per_k(ts, HEDGES_UNI, per, min_tokens),
            "has_profane":   _has_any(ts, PROFANITY),
            "has_identity":  _has_any(ts, IDENTITY),
            "has_hedge_bigram": _has_any_bigram(ts, HEDGE_PHRASES),
            "n_tokens":      len(ts),
            "avg_tok_len":   float(np.mean([len(t) for t in ts])) if ts else 0.0,
            **(lambda ru: dict(zip(["had_url","had_mention","hashtag_count","bangs","qmarks"], ru)))(_raw_counts(df.iloc[i][raw_col] if raw_col in df.columns else "")),
        })
    feats = pd.DataFrame(rows, index=df.index)
    return feats[CUE_COLS]

# -----------------------------
# Cue scaling and alignment
# -----------------------------
def fit_cue_scaler(train_df: pd.DataFrame) -> MaxAbsScaler:
    """Fit MaxAbs on train cue DataFrame (no centering)."""
    sc = MaxAbsScaler()
    sc.fit(train_df.values)
    return sc

def transform_cues(df: pd.DataFrame, scaler: MaxAbsScaler, cue_order: Optional[List[str]] = None) -> csr_matrix:
    """
    Transform a cue DataFrame with a fitted MaxAbs scaler.
    If cue_order is provided, columns are aligned before transform.
    Returns CSR sparse matrix.
    """
    X = df[cue_order].values if cue_order is not None else df.values
    return csr_matrix(scaler.transform(X))

def save_cue_artifacts(scaler: MaxAbsScaler, cue_names: List[str], name_prefix: str) -> None:
    """Save cue scaler and column order under artifacts/<prefix>_cues_(scaler|columns).joblib"""
    joblib.dump(scaler, artifact(f"{name_prefix}_cues_scaler.joblib"))
    joblib.dump(cue_names, artifact(f"{name_prefix}_cue_columns.joblib"))

def load_cue_artifacts(name_prefix: str) -> Tuple[MaxAbsScaler, List[str]]:
    """Load cue scaler and column order previously saved with save_cue_artifacts."""
    sc = joblib.load(artifact(f"{name_prefix}_cues_scaler.joblib"))
    cols = joblib.load(artifact(f"{name_prefix}_cue_columns.joblib"))
    return sc, cols

def rescale_for_transfer(
    X_src_scaled: csr_matrix, src_scaler: MaxAbsScaler, tgt_scaler: MaxAbsScaler
) -> csr_matrix:
    """
    Cross-domain helper: invert source scaling to raw, then apply target scaler.
    Example: TX cues scaled with TX scaler → back to raw → scale with CB scaler.
    """
    Xraw = src_scaler.inverse_transform(X_src_scaled.toarray())
    return csr_matrix(tgt_scaler.transform(Xraw))

# -----------------------------
# TF–IDF helpers
# -----------------------------
def fit_word_tfidf(
    train_texts: pd.Series,
    ngram_range: Tuple[int,int] = (1,2),
    min_df: int = 3,
    max_features: int = 50_000
) -> TfidfVectorizer:
    """Fit word-level TF–IDF on train texts only."""
    vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_features=max_features)
    vec.fit(train_texts)
    return vec

def fit_char_tfidf(
    train_texts: pd.Series,
    ngram_range: Tuple[int,int] = (3,5),
    min_df: int = 5,
    max_features: int = 30_000
) -> TfidfVectorizer:
    """Fit character-level TF–IDF (for obfuscations) on train texts only."""
    vec = TfidfVectorizer(analyzer="char", ngram_range=ngram_range, min_df=min_df, max_features=max_features)
    vec.fit(train_texts)
    return vec

def tfidf_transform(vec: TfidfVectorizer, texts: pd.Series) -> csr_matrix:
    """Transform texts with a fitted TF–IDF vectorizer (word or char)."""
    return vec.transform(texts)

def save_vectorizer(vec: TfidfVectorizer, name: str) -> None:
    """Save a fitted TfidfVectorizer to artifacts (e.g., 'tfidf_word_cyberbullying.joblib')."""
    joblib.dump(vec, artifact(name))

def load_vectorizer(name: str) -> TfidfVectorizer:
    """Load a fitted TfidfVectorizer from artifacts."""
    return joblib.load(artifact(name))

# -----------------------------
# Feature assembly
# -----------------------------
def stack_features(
    X_word: csr_matrix,
    X_char: csr_matrix,
    X_cues: csr_matrix
) -> Dict[str, csr_matrix]:
    """
    Convenience to horizontally stack 3 blocks for a single split.
    In notebooks you’ll typically call this per split and save the dict yourself.
    """
    return {"X": hstack([X_word, X_char, X_cues], format="csr")}

__all__ = [
    # cue building
    "build_psych_features_smooth", "CUE_COLS",
    # scaling
    "fit_cue_scaler", "transform_cues", "save_cue_artifacts", "load_cue_artifacts", "rescale_for_transfer",
    # tfidf
    "fit_word_tfidf", "fit_char_tfidf", "tfidf_transform", "save_vectorizer", "load_vectorizer",
    # assembly
    "stack_features",
    # lexicons (optional export)
    "PROFANITY","VIOLENCE","PRON_2ND","INTENS","POLITE","NEGATE","IDENTITY","GENERALIZ","HEDGES_UNI","HEDGE_PHRASES"
]
