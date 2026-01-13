"""Semantic diff using embeddings."""

import numpy as np
from typing import Optional, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

# Similarity thresholds
SIMILARITY_THRESHOLD = 0.8
VERY_SIMILAR_THRESHOLD = 0.95


def embed(text: str, model: Optional[str] = None) -> np.ndarray:
    """
    Generate embedding for text.
    
    Args:
        text: Text to embed
        model: Embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)
    
    Returns:
        Embedding vector as numpy array
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        # Fallback to a simple hash-based embedding if sentence-transformers not available
        # This is not ideal but allows the package to work without extra dependencies
        return _simple_embed(text)
    
    model_name = model or "sentence-transformers/all-MiniLM-L6-v2"
    
    # Cache the model to avoid reloading
    if not hasattr(embed, "_model_cache"):
        embed._model_cache = {}
    
    if model_name not in embed._model_cache:
        embed._model_cache[model_name] = SentenceTransformer(model_name)
    
    transformer = embed._model_cache[model_name]
    embedding = transformer.encode(text, convert_to_numpy=True)
    
    # Reshape to 2D if needed
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    
    return embedding


def _simple_embed(text: str) -> np.ndarray:
    """
    Simple fallback embedding using hash-based approach.
    This is a poor substitute for real embeddings but allows basic functionality.
    """
    import hashlib
    
    # Create a simple hash-based vector
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()
    
    # Convert to float array and normalize
    vector = np.frombuffer(hash_bytes[:16], dtype=np.uint8).astype(np.float32)
    vector = vector / np.linalg.norm(vector) if np.linalg.norm(vector) > 0 else vector
    
    return vector.reshape(1, -1)


def semantic_diff(a: str, b: str, embedding_model: Optional[str] = None) -> float:
    """
    Calculate semantic similarity between two texts using embeddings.
    
    Args:
        a: Baseline text
        b: Candidate text
        embedding_model: Optional embedding model name
    
    Returns:
        Cosine similarity score between 0 and 1
    """
    ea = embed(a, model=embedding_model)
    eb = embed(b, model=embedding_model)
    
    # Ensure same dimensions
    if ea.shape[1] != eb.shape[1]:
        # Pad or truncate to match
        min_dim = min(ea.shape[1], eb.shape[1])
        ea = ea[:, :min_dim]
        eb = eb[:, :min_dim]
    
    similarity = cosine_similarity(ea, eb)[0][0]
    return float(similarity)


def semantic_diff_detailed(a: str, b: str, embedding_model: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate detailed semantic similarity metrics.
    
    Args:
        a: Baseline text
        b: Candidate text
        embedding_model: Optional embedding model name
    
    Returns:
        Dictionary with similarity metrics
    """
    similarity = semantic_diff(a, b, embedding_model)
    
    return {
        "similarity": similarity,
        "baseline_length": len(a),
        "candidate_length": len(b),
        "is_similar": similarity > SIMILARITY_THRESHOLD,
        "is_very_similar": similarity > VERY_SIMILAR_THRESHOLD
    }
