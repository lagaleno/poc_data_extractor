# app/extractors/base.py
from typing import List, Dict

class ExtractorBase:
    name: str = "base"

    def __init__(self, rqs: List[str], examples: List[Dict]):
        """
        rqs: lista de perguntas (em inglês)
        examples: lista com 2 exemplos: [{"file_bytes": bytes|None, "answers": [str,...]}, ...]
        """
        self.rqs = rqs
        self.examples = examples

    def extract(self, article_text: str) -> List[str]:
        """Retorna respostas (lista alinhada às RQs)."""
        raise NotImplementedError
