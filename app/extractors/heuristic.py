# app/extractors/heuristic.py
import re
from typing import List, Dict
from .base import ExtractorBase

EN_STOP = {
    "the","a","an","of","and","or","to","for","with","in","on","as","is","are","was","were",
    "this","that","these","those","by","from","at","be","been","it","its","their","they",
    "we","our","you","your","not","no","yes","into","about","over","under","than","then",
    "which","who","whom","whose","what","when","where","why","how"
}

def _normalize_tokens(s: str):
    tokens = re.findall(r"[A-Za-z0-9]+", s.lower())
    return [t for t in tokens if t not in EN_STOP and len(t) > 1]

def _split_sentences(text: str):
    text = re.sub(r"\s+", " ", text)
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    if len(sents) < 2:
        sents = re.split(r"[.\n]", text)
    return [s.strip() for s in sents if s and len(s.strip()) > 2]

def _score(sentence: str, query: str) -> float:
    s_tokens = set(_normalize_tokens(sentence))
    q_tokens = set(_normalize_tokens(query))
    if not s_tokens or not q_tokens:
        return 0.0
    return len(s_tokens & q_tokens) / len(s_tokens | q_tokens)

def _heuristic_answers(text: str, rqs: List[str]) -> List[str]:
    sentences = _split_sentences(text)
    answers = []
    lower = text.lower()
    abs_pos = lower.find("abstract")
    for rq in rqs:
        rq = (rq or "").strip()
        if not rq:
            answers.append("")
            continue
        best_score, best_sent = 0.0, ""
        for s in sentences:
            sc = _score(s, rq)
            if sc > best_score:
                best_score, best_sent = sc, s
        if best_score == 0.0:
            snippet = text[abs_pos:abs_pos+300] if abs_pos != -1 else text[:300]
            answers.append(snippet.strip())
        else:
            answers.append(best_sent)
    return answers

class HeuristicExtractor(ExtractorBase):
    name: str = "heuristic"

    def extract(self, article_text: str) -> List[str]:
        ans = _heuristic_answers(article_text, self.rqs)
        # normaliza tamanho
        if len(ans) < len(self.rqs):
            ans += [""] * (len(self.rqs) - len(ans))
        return ans[:len(self.rqs)]
