# app/extractors/__init__.py
from typing import List, Dict
from .base import ExtractorBase
from .heuristic import HeuristicExtractor

def make_extractor(modelo_ui: str, rqs: List[str], examples: List[Dict]) -> ExtractorBase:
    """
    modelo_ui: o texto vindo do selectbox ("OpenAI", "LLaMA", "Gemini", "Heurístico (Regex)")
    Por enquanto implementamos OpenAI e Heurístico; LLaMA/Gemini poderiam ser adicionados depois.
    """
    name = (modelo_ui or "").strip().lower()
    if "openai" in name:
        try:
            from .openai_extractor import OpenAIExtractor
            return OpenAIExtractor(rqs=rqs, examples=examples)
        except Exception as e:
            # fallback se faltar lib/chave
            import streamlit as st
            st.warning(f"OpenAI extractor unavailable ({e}). Falling back to heuristic.")
            return HeuristicExtractor(rqs=rqs, examples=examples)

    # TODO: adicionar GeminiExtractor / LLaMAExtractor futuramente
    return HeuristicExtractor(rqs=rqs, examples=examples)
