# OpenAI extractor placeholder
# app/extractors/openai_extractor.py
from typing import List, Dict
import os, json
from .base import ExtractorBase

class OpenAIExtractor(ExtractorBase):
    name: str = "openai"

    def __init__(self, rqs: List[str], examples: List[Dict], model: str = "gpt-4o-mini"):
        super().__init__(rqs, examples)
        try:
            from openai import OpenAI  # import lazy
        except Exception as e:
            raise RuntimeError("OpenAI SDK not installed. Run: pip install openai") from e
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing in environment.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    @staticmethod
    def _build_shots(examples: List[Dict], rqs: List[str]):
        # Em vez de carregar PDFs aqui, assumimos que o app já extraiu o texto;
        # para simplificar, usamos apenas (rq, answer) dos exemplos.
        shots = []
        for ex in examples:
            pairs = []
            for i, rq in enumerate(rqs, start=1):
                rq_clean = (rq or "").strip()
                if not rq_clean:
                    continue
                ans = ex["answers"][i-1] if i-1 < len(ex["answers"]) else ""
                pairs.append({"rq": rq_clean, "answer": ans})
            if pairs:
                shots.append({"pairs": pairs})
        return shots

    def extract(self, article_text: str) -> List[str]:
        system_msg = (
            "You answer Software Engineering research questions from the text of a scientific article. "
            "Return ONLY valid JSON: a list of strings, in the SAME ORDER as the input RQs. "
            "When unsure, return an empty string. Do not fabricate facts."
        )
        user_payload = {
            "instruction": "All content is in ENGLISH. Use the provided examples as weak supervision (few-shot).",
            "rqs": self.rqs,
            "examples": self._build_shots(self.examples, self.rqs),
            "article_text": article_text[:12000]
        }
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
            ]
        )
        content = resp.choices[0].message.content
        try:
            answers = json.loads(content)
            if not isinstance(answers, list):
                answers = []
        except Exception:
            answers = []

        # normalize length
        answers = [(a if isinstance(a, str) else str(a)) for a in answers]
        if len(answers) < len(self.rqs):
            answers += [""] * (len(self.rqs) - len(answers))
        return answers[:len(self.rqs)]