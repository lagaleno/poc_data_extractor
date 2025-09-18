# app/extractors/gemini_extractor.py
from typing import List, Dict
import os, json, re
from .base import ExtractorBase

def _ensure_len(values: List[str], n: int) -> List[str]:
    values = [(v if isinstance(v, str) else str(v)) for v in (values or [])]
    if len(values) < n:
        values += [""] * (n - len(values))
    return values[:n]

def _extract_json_maybe(text: str):
    """Tenta obter uma lista JSON mesmo se o modelo 'falar demais'."""
    if not text:
        return None
    # remove cercas de código
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    # 1) tentativa direta
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) primeiro bloco entre colchetes
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # 3) dicionário com chave 'answers'
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict) and isinstance(data.get("answers"), list):
                return data["answers"]
        except Exception:
            pass
    return None

class GeminiExtractor(ExtractorBase):
    """
    Usa google-generativeai (Gemini). Requer:
      - GOOGLE_API_KEY no ambiente
    Modelo padrão: gemini-1.5-flash (rápido e barato). Troque se quiser.
    """
    name: str = "gemini"

    def __init__(self, rqs: List[str], examples: List[Dict], model: str = "gemini-1.5-flash"):
        super().__init__(rqs, examples)
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as e:
            raise RuntimeError("google-generativeai não instalada. Rode: pip install google-generativeai") from e
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY não encontrada no ambiente.")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model

    def extract(self, article_text: str) -> List[str]:
        model = self._genai.GenerativeModel(self._model_name)

        prompt = (
            "You answer Software Engineering research questions from the text of a scientific article.\n"
            "Return ONLY valid JSON: a list of strings, in the SAME ORDER as the input RQs.\n"
            "When unsure, return an empty string. Do not fabricate facts.\n\n"
            f"RQs (English): {self.rqs}\n\n"
            "Article text (English, may contain OCR noise):\n"
            f"{article_text[:12000]}"
        )

        try:
            res = model.generate_content(prompt)
            # Em algumas versões, o texto vem em res.text; em outras, usar candidates/parts
            text = getattr(res, "text", None)
            print(text)
            if text is None:
                # fallback para estrutura crua
                try:
                    cand = (res.candidates or [])[0]
                    parts = getattr(cand.content, "parts", None) or []
                    text = "".join(getattr(p, "text", "") for p in parts)
                except Exception:
                    text = ""

            parsed = _extract_json_maybe(text)
            if isinstance(parsed, list):
                return _ensure_len(parsed, len(self.rqs))
            return _ensure_len([], len(self.rqs))
        except Exception as e:
            raise RuntimeError(f"Gemini request failed: {e}") from e
