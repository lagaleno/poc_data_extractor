# app/extractors/__init__.py
from typing import List, Dict
from .base import ExtractorBase
from .heuristic import HeuristicExtractor

def make_extractor(modelo_ui: str, rqs: List[str], examples: List[Dict]) -> ExtractorBase:
    name = (modelo_ui or "").strip().lower()

    if "openai" in name:
        try:
            from .openai_extractor import OpenAIExtractor
            return OpenAIExtractor(rqs=rqs, examples=examples)
        except Exception as e:
            try:
                import streamlit as st
                st.warning(f"OpenAI extractor unavailable ({e}). Falling back to heuristic.")
            except Exception:
                pass
            return HeuristicExtractor(rqs=rqs, examples=examples)

    if "gemini" in name:
        try:
            from .gemini_extractor import GeminiExtractor
            return GeminiExtractor(rqs=rqs, examples=examples)
        except Exception as e:
            try:
                import streamlit as st
                st.warning(f"Gemini extractor unavailable ({e}). Falling back to heuristic.")
            except Exception:
                pass
            return HeuristicExtractor(rqs=rqs, examples=examples)


        try:
            from .llama_extractor import LLaMAExtractor
            return LLaMAExtractor(rqs=rqs, examples=examples)
        except Exception as e:
            try:
                import streamlit as st
                st.warning(f"LLaMA extractor unavailable ({e}). Falling back to heuristic.")
            except Exception:
                pass
            return HeuristicExtractor(rqs=rqs, examples=examples)

    return HeuristicExtractor(rqs=rqs, examples=examples)
