# app/extractors/heuristic.py
from typing import List, Sequence
import re
import numpy as np

from .base import ExtractorBase

# =========================
# NLTK bootstrap silencioso
# =========================
def _ensure_nltk():
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

def _sent_tokenize(text: str) -> List[str]:
    import nltk
    from nltk.tokenize import sent_tokenize
    text = re.sub(r"\s+", " ", text).strip()
    sents = sent_tokenize(text)
    return [s.strip() for s in sents if len(s.strip()) > 2]

# ================
# TF-IDF utilities
# ================
def _build_vectorizer(stop_words):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(
        lowercase=True,
        token_pattern=r"[A-Za-z0-9]+",
        stop_words=stop_words,
        ngram_range=(1, 2),
        min_df=1,
    )

def _cosine_sim_matrix(A, B):
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(A, B)

# ==========================
# Embeddings (opcionais)
# ==========================
def _embed_sentences(sentences: Sequence[str]):
    """Retorna (model, embeddings np.array). Lança ImportError se faltam deps."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(list(sentences), convert_to_numpy=True, normalize_embeddings=True)
    return model, emb

def _embed_query(model, texts: Sequence[str]):
    """Usa o mesmo modelo para codificar queries (normalizado)."""
    emb = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
    return emb

# ==========================
# HeuristicExtractor avançado
# ==========================
class HeuristicExtractor(ExtractorBase):
    """
    Heurístico com combinação de similaridade:
      score = w_rq * sim(sent, RQ) + w_ex * mean(sim(sent, example_answers_for_RQ))

    - Usa embeddings (sentence-transformers) se disponíveis; senão TF-IDF + cosseno.
    - Concatena top-k sentenças para formar a resposta.
    """
    name: str = "heuristic"

    def __init__(self, rqs: List[str], examples):
        super().__init__(rqs, examples)
        _ensure_nltk()
        from nltk.corpus import stopwords
        self._stop = set(stopwords.words("english"))

        # Hiperparâmetros
        self.top_k = 2        # quantas sentenças concatenar por RQ
        self.w_rq = 0.7       # peso da similaridade com a RQ
        self.w_ex = 0.3       # peso da similaridade com respostas de exemplo

    def _examples_for_rq(self, rq_idx: int) -> List[str]:
        """Coleta as respostas de exemplo correspondentes à RQ de índice rq_idx (0-based)."""
        res = []
        for ex in (self.examples or []):
            ans_list = ex.get("answers") if isinstance(ex, dict) else None
            if isinstance(ans_list, list) and rq_idx < len(ans_list):
                ans = (ans_list[rq_idx] or "").strip()
                if ans:
                    res.append(ans)
        return res

    def _rank_with_tfidf(self, rq: str, ex_answers: List[str], sentences: List[str]) -> List[int]:
        """
        TF-IDF ranking combinando:
          sim_rq = cos( tfidf(rq), tfidf(sent_i) )
          sim_ex = mean_i cos( tfidf(ex_i), tfidf(sent_i) )
          score  = w_rq * sim_rq + w_ex * sim_ex
        """
        from scipy.sparse import vstack

        # Monta corpus: [rq] + ex_answers + sentences
        corpus = [rq] + ex_answers + sentences if ex_answers else [rq] + sentences
        vectorizer = _build_vectorizer(self._stop)
        X = vectorizer.fit_transform(corpus)

        n_ex = len(ex_answers)
        q_vec = X[0:1]
        if n_ex:
            ex_vecs = X[1:1+n_ex]
            sent_vecs = X[1+n_ex:]
        else:
            ex_vecs = None
            sent_vecs = X[1:]

        # Similaridade com RQ
        sim_rq = _cosine_sim_matrix(q_vec, sent_vecs).ravel()  # shape: (n_sent,)

        # Similaridade com exemplos (média)
        if ex_vecs is not None and ex_vecs.shape[0] > 0:
            sims_ex = _cosine_sim_matrix(ex_vecs, sent_vecs)    # shape: (n_ex, n_sent)
            sim_ex = sims_ex.mean(axis=0).ravel()
        else:
            sim_ex = np.zeros_like(sim_rq)

        score = self.w_rq * sim_rq + self.w_ex * sim_ex
        return list(np.argsort(-score))

    def _rank_with_embeddings(self, rq: str, ex_answers: List[str], sentences: List[str]) -> List[int]:
        """
        Embeddings ranking combinando:
          sim_rq = cos( emb(rq), emb(sent_i) )
          sim_ex = mean_i cos( emb(ex_i), emb(sent_i) )
        """
        # Codifica tudo em um lote
        items = [rq] + ex_answers + sentences if ex_answers else [rq] + sentences
        model, all_emb = _embed_sentences(items)

        n_ex = len(ex_answers)
        q_emb = all_emb[0:1, :]
        if n_ex:
            ex_embs = all_emb[1:1+n_ex, :]
            sent_embs = all_emb[1+n_ex:, :]
        else:
            ex_embs = None
            sent_embs = all_emb[1:, :]

        # cos sim manual (já normalizado)
        sim_rq = (q_emb @ sent_embs.T).ravel()  # (n_sent,)

        if ex_embs is not None and ex_embs.shape[0] > 0:
            sims_ex = ex_embs @ sent_embs.T     # (n_ex, n_sent)
            sim_ex = sims_ex.mean(axis=0).ravel()
        else:
            sim_ex = np.zeros_like(sim_rq)

        score = self.w_rq * sim_rq + self.w_ex * sim_ex
        return list(np.argsort(-score))

    def extract(self, article_text: str) -> List[str]:
        # 1) Divide o artigo em sentenças
        sentences = _sent_tokenize(article_text or "")
        if not sentences:
            return [""] * len(self.rqs)

        answers: List[str] = []

        for i, rq in enumerate(self.rqs):
            rq_clean = (rq or "").strip()
            if not rq_clean:
                answers.append("")
                continue

            ex_answers = self._examples_for_rq(i)

            # 2) Ranking com embeddings (se disponíveis); senão TF-IDF
            try:
                rank = self._rank_with_embeddings(rq_clean, ex_answers, sentences)
            except Exception:
                rank = self._rank_with_tfidf(rq_clean, ex_answers, sentences)

            # 3) Seleciona top-k e concatena
            if not rank:
                answers.append("")
                continue
            top_idxs = rank[: self.top_k]
            chosen = " ".join(sentences[j] for j in top_idxs if 0 <= j < len(sentences)).strip()

            # 4) Fallback se vazio: Abstract/beginning
            if not chosen:
                lower = (article_text or "").lower()
                pos = lower.find("abstract")
                snippet = (article_text or "")[pos:pos+300] if pos != -1 else (article_text or "")[:300]
                chosen = snippet.strip()

            answers.append(chosen)

        # Normaliza comprimento
        if len(answers) < len(self.rqs):
            answers += [""] * (len(self.rqs) - len(answers))
        return answers[:len(self.rqs)]
