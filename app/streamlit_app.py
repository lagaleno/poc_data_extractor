# app/streamlit_app.py
import streamlit as st

# ==============================
# Helpers da Etapa 3 (coloque no topo do arquivo se ainda não existirem)
# ==============================
# ===== EN-only helpers (replace the previous helpers) =====
import io, re, time
import pdfplumber
import pandas as pd
import json

EN_STOP = {
    "the","a","an","of","and","or","to","for","with","in","on","as","is","are","was","were",
    "this","that","these","those","by","from","at","be","been","it","its","their","they",
    "we","our","you","your","not","no","yes","into","about","over","under","than","then",
    "which","who","whom","whose","what","when","where","why","how"
}

def extract_text_from_pdf(file_bytes: bytes, max_pages: int | None = None) -> str:
    text_chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
        for p in pages:
            try:
                text_chunks.append(p.extract_text() or "")
            except Exception:
                pass
    return "\n".join(text_chunks)

def _normalize_tokens(s: str):
    tokens = re.findall(r"[A-Za-z0-9]+", s.lower())
    return [t for t in tokens if t not in EN_STOP and len(t) > 1]

def _split_sentences(text: str):
    # Simple English-centric sentence split
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

def heuristic_answers(text: str, rqs: list[str]) -> list[str]:
    """
    Very simple heuristic baseline for English PDFs:
    picks the sentence with the best token-overlap score per RQ;
    if nothing matches, returns a snippet from the Abstract or beginning.
    """
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

def run_extraction(modelo: str, full_text: str, rqs: list[str], examples: list[dict]) -> list[str]:
    """
    Route extraction by selected model.
    All prompts assume ENGLISH inputs (PDF, research objective, RQs, answers).
    Always returns a list of strings aligned with RQs length.
    """
    # Always ensure we return the right length
    def _pad(lst: list[str], n: int) -> list[str]:
        lst = [(x if isinstance(x, str) else str(x)) for x in (lst or [])]
        if len(lst) < n: lst += [""] * (n - len(lst))
        return lst[:n]

    # 1) Heuristic baseline (English)
    if modelo == "Heurístico (Regex)":
        return _pad(heuristic_answers(full_text, rqs), len(rqs))

    # Build few-shot examples (short)
    def build_shots():
        shots = []
        for ex in examples:
            if not ex.get("file_bytes"):
                continue
            ex_text = extract_text_from_pdf(ex["file_bytes"], max_pages=2)
            pairs = []
            for i, rq in enumerate(rqs, start=1):
                rq_clean = (rq or "").strip()
                if not rq_clean:
                    continue
                ans = ex["answers"][i-1] if i-1 < len(ex["answers"]) else ""
                pairs.append({"rq": rq_clean, "answer": ans})
            shots.append({"context": ex_text[:4000], "pairs": pairs})
        return shots

    # 2) OpenAI
    if modelo == "OpenAI":
        try:
            import os, json
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            system_msg = (
                "You answer Software Engineering research questions from the text of a scientific article. "
                "Return ONLY valid JSON: a list of strings, in the SAME ORDER as the input RQs. "
                "When unsure, return an empty string. Do not fabricate facts."
            )
            user_payload = {
                "instruction": "All content is in ENGLISH. Use the examples as weak supervision (few-shot).",
                "rqs": rqs,
                "examples": build_shots(),
                "article_text": full_text[:12000]
            }
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                ]
            )
            content = resp.choices[0].message.content
            answers = json.loads(content)
            if isinstance(answers, list):
                return _pad(answers, len(rqs))
        except Exception as e:
            st.warning(f"OpenAI unavailable ({e}). Falling back to heuristic.")
            return _pad(heuristic_answers(full_text, rqs), len(rqs))

    # 3) Gemini
    if modelo == "Gemini":
        try:
            import os, json, google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")

            prompt = (
                "You answer Software Engineering research questions from the text of a scientific article.\n"
                "Return ONLY valid JSON: a list of strings, in the SAME ORDER as the input RQs.\n"
                "When unsure, return an empty string. Do not fabricate facts.\n\n"
                f"RQs (English): {rqs}\n\n"
                "Article text (English, may contain OCR noise):\n"
                f"{full_text[:12000]}"
            )
            res = model.generate_content(prompt)
            answers = json.loads(res.text)
            if isinstance(answers, list):
                return _pad(answers, len(rqs))
        except Exception as e:
            st.warning(f"Gemini unavailable ({e}). Falling back to heuristic.")
            return _pad(heuristic_answers(full_text, rqs), len(rqs))

    # 4) LLaMA (via Ollama local)
    if modelo == "LLaMA":
        try:
            import json, requests
            prompt = (
                "You answer Software Engineering research questions from the text of a scientific article.\n"
                "Return ONLY valid JSON: a list of strings, in the SAME ORDER as the input RQs.\n"
                "When unsure, return an empty string. Do not fabricate facts.\n\n"
                f"RQs (English): {rqs}\n\n"
                "Article text (English, may contain OCR noise):\n"
                f"{full_text[:8000]}"
            )
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3", "prompt": prompt, "stream": False},
                timeout=60
            )
            data = resp.json()
            answers = json.loads(data.get("response", "[]"))
            if isinstance(answers, list):
                return _pad(answers, len(rqs))
        except Exception as e:
            st.warning(f"LLaMA (Ollama) unavailable ({e}). Falling back to heuristic.")
            return _pad(heuristic_answers(full_text, rqs), len(rqs))

    # Final fallback
    return _pad(heuristic_answers(full_text, rqs), len(rqs))



# ==============================
# Configurações iniciais
# ==============================
st.set_page_config(page_title="PoC Extrator de Artigos PDF", layout="wide")

st.title("📄 PoC Extrator de Artigos PDF")
st.markdown("Auxílio à extração de dados de artigos em **estudos secundários**")

# ==============================
# Estado inicial (navegação por etapas)
# ==============================
if "stage" not in st.session_state:
    st.session_state.stage = "step1"  # step1 -> step2 -> (step3 em breve)

if "examples" not in st.session_state:
    # Lista com 2 exemplos; cada exemplo terá: {"filename": str|None, "file_bytes": bytes|None, "answers": [str,...]}
    st.session_state.examples = [
        {"filename": None, "file_bytes": None, "answers": []},
        {"filename": None, "file_bytes": None, "answers": []},
    ]

# ==============================
# Sidebar - Configurações e link de navegação
# ==============================
st.sidebar.header("⚙️ Configurações")
modelo = st.sidebar.selectbox(
    "Selecione o modelo de IA",
    ["OpenAI", "LLaMA", "Gemini", "Heurístico (Regex)"],
    index=0
)

st.sidebar.info(f"🔍 Modelo selecionado: **{modelo}**")

# st.sidebar.markdown("---")
# st.sidebar.subheader("🔎 Navegação")
# try:
#    st.sidebar.page_link("pages/1_📌_Estado_atual.py", label="📌 Estado atual", icon="📌")
# except Exception:
#    st.sidebar.info("Vá em **Páginas » 📌 Estado atual** no menu lateral.")

# ==============================
# Etapa 1 - Objetivo e perguntas
# ==============================
st.header("1️⃣ Objetivo da pesquisa e Perguntas de Pesquisa")
st.info("🔤 **Important:** Although the UI is in Portuguese, please enter the *research objective*, *RQs*, and upload articles **in ENGLISH**. All extraction prompts and processing assume English.")


# Guardar objetivo também no estado, para reaparecer se navegar entre etapas
if "objetivo" not in st.session_state:
    st.session_state.objetivo = ""

objetivo = st.text_area(
    "Qual o **objetivo da pesquisa**?",
    value=st.session_state.objetivo,
    placeholder="Digite aqui o objetivo geral da sua revisão...",
    key="objetivo_input"
)

# Sincroniza objetivo no estado sempre que o usuário editar
st.session_state.objetivo = objetivo

st.subheader("Perguntas de Pesquisa (RQs)")
if "rqs" not in st.session_state:
    st.session_state.rqs = [""]  # começa com 1 campo

# Form com dois botões: Adicionar e Salvar
with st.form("perguntas_form", clear_on_submit=False):
    novas_rqs = []
    for i, rq in enumerate(st.session_state.rqs):
        novas_rqs.append(
            st.text_input(f"RQ{i+1}", key=f"rq_{i}", value=rq, placeholder="Descreva a pergunta de pesquisa")
        )

    c1, c2 = st.columns(2)
    with c1:
        add = st.form_submit_button("➕ Adicionar Pergunta")
    with c2:
        save = st.form_submit_button("💾 Salvar Perguntas")

    if add:
        st.session_state.rqs = novas_rqs + [""]
        st.rerun()

    if save:
        st.session_state.rqs = [rq.strip() for rq in novas_rqs]
        st.success("Perguntas salvas!")

# Exibir estado atual resumido na página principal
if st.session_state.objetivo.strip():
    st.success(f"🎯 Objetivo registrado: {st.session_state.objetivo}")

if st.session_state.rqs and any(rq.strip() for rq in st.session_state.rqs):
    st.info("📌 Perguntas registradas:")
    for idx, rq in enumerate(st.session_state.rqs, start=1):
        if rq.strip():
            st.write(f"- **RQ{idx}:** {rq}")

# ==============================
# Gate para avançar para a Etapa 2
# ==============================
ready_for_examples = bool(st.session_state.objetivo.strip()) and any(rq.strip() for rq in st.session_state.rqs)

st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    if ready_for_examples:
        if st.button("➡️ Ir para a etapa 2 — Exemplos", use_container_width=True):
            st.session_state.stage = "step2"
            st.rerun()
    else:
        st.button("➡️ Ir para a etapa 2 — Exemplos", disabled=True, use_container_width=True)
with col_b:
    st.caption("Preencha **objetivo** e pelo menos **1 RQ** para habilitar a próxima etapa.")

# ==============================
# Etapa 2 - Exemplos (2 artigos + respostas por RQ)
# ==============================
if st.session_state.stage == "step2":
    st.header("2️⃣ Exemplos — 2 artigos e respostas por RQ")

    if not ready_for_examples:
        st.warning("Para continuar, volte e preencha o **objetivo** e ao menos **uma pergunta de pesquisa**.")
    else:
        st.markdown(
            "Faça upload de **2 artigos de exemplo** (PDF). Para cada exemplo, "
            "preencha **as respostas** às suas RQs (isso servirá como *few-shot* para orientar a extração)."
        )

        # Garantir que o vetor answers de cada exemplo tenha mesmo tamanho que RQs
        num_rqs = len(st.session_state.rqs)
        for ex in st.session_state.examples:
            if len(ex['answers']) != num_rqs:
                ex['answers'] = (ex['answers'] + [""] * num_rqs)[:num_rqs]

        with st.form("examples_form", clear_on_submit=False):
            # EXEMPLO 1
            st.subheader("📘 Exemplo 1")
            f1 = st.file_uploader("Artigo PDF (Exemplo 1)", type=["pdf"], key="ex1_pdf")
            if f1 is not None:
                st.session_state.examples[0]["filename"] = f1.name
                st.session_state.examples[0]["file_bytes"] = f1.getvalue()

            # Respostas do Exemplo 1
            for i, rq in enumerate(st.session_state.rqs, start=1):
                if rq.strip():
                    st.session_state.examples[0]["answers"][i-1] = st.text_area(
                        f"Resposta para RQ{i}: {rq}",
                        value=st.session_state.examples[0]["answers"][i-1],
                        key=f"ex1_rq_{i}",
                        height=100,
                        placeholder="Digite a resposta do exemplo 1 para esta RQ"
                    )

            st.markdown("---")

            # EXEMPLO 2
            st.subheader("📗 Exemplo 2")
            f2 = st.file_uploader("Artigo PDF (Exemplo 2)", type=["pdf"], key="ex2_pdf")
            if f2 is not None:
                st.session_state.examples[1]["filename"] = f2.name
                st.session_state.examples[1]["file_bytes"] = f2.getvalue()

            # Respostas do Exemplo 2
            for i, rq in enumerate(st.session_state.rqs, start=1):
                if rq.strip():
                    st.session_state.examples[1]["answers"][i-1] = st.text_area(
                        f"Resposta para RQ{i}: {rq}",
                        value=st.session_state.examples[1]["answers"][i-1],
                        key=f"ex2_rq_{i}",
                        height=100,
                        placeholder="Digite a resposta do exemplo 2 para esta RQ"
                    )

            st.markdown("---")

            c1, c2, c3 = st.columns(3)
            with c1:
                back = st.form_submit_button("⬅️ Voltar para Etapa 1")
            with c2:
                save = st.form_submit_button("💾 Salvar exemplos")
            with c3:
                next_step = st.form_submit_button("➡️ Continuar para Etapa 3 (extração)")

        # Ações pós-submit
        if 'back' in locals() and back:
            st.session_state.stage = "step1"
            st.rerun()

        if 'save' in locals() and save:
            # Validação simples
            missing = []
            for idx, ex in enumerate(st.session_state.examples, start=1):
                if not ex["file_bytes"]:
                    missing.append(f"Exemplo {idx}: arquivo PDF")
                for i, rq in enumerate(st.session_state.rqs, start=1):
                    if rq.strip() and not ex["answers"][i-1].strip():
                        missing.append(f"Exemplo {idx}: resposta para RQ{i}")
            if missing:
                st.error("⚠️ Preencha os itens obrigatórios:\n\n- " + "\n- ".join(missing))
            else:
                st.success("✅ Exemplos salvos com sucesso!")

        if 'next_step' in locals() and next_step:
            # Apenas checagem antes de seguir (vamos implementar extração na etapa 3 futuramente)
            ok = True
            for idx, ex in enumerate(st.session_state.examples, start=1):
                if not ex["file_bytes"]:
                    ok = False
                    st.error(f"⚠️ Falta o PDF do **Exemplo {idx}**.")
            if ok:
                st.session_state.stage = "step3"
                st.info("Próxima etapa (extração) será implementada no próximo passo.")


# ==============================
# Etapa 3 — Extração (1 artigo por vez, acumulando resultados)
# ==============================
if "extracted_rows" not in st.session_state:
    st.session_state.extracted_rows = []  # cada item é um dict com {filename, timestamp, rq1, rq2, ...}

# contador para resetar o file_uploader quando quisermos extrair outro
if "upload_counter" not in st.session_state:
    st.session_state.upload_counter = 0

if st.session_state.stage == "step3":
    st.header("3️⃣ Extração automática — 1 artigo por vez")

    # Pré-condição: já ter objetivo + ao menos 1 RQ
    if not (st.session_state.get("objetivo","").strip() and any(rq.strip() for rq in st.session_state.get("rqs",[]))):
        st.warning("Preencha o **objetivo** e ao menos **uma RQ** na Etapa 1 antes de extrair.")
    else:
        st.markdown("Faça upload de **um PDF** por vez. Você pode repetir o processo quantas vezes quiser; os resultados serão acumulados na tabela abaixo.")

        # use uma key variável para permitir "resetar" o uploader ao clicar em 'Extrair outro PDF'
        f = st.file_uploader(
            "Artigo PDF para extração",
            type=["pdf"],
            key=f"extract_pdf_{st.session_state.upload_counter}",
            help="Suba um PDF por vez. Após extrair, use 'Extrair outro PDF' para processar o próximo."
        )
        if f is not None:
            st.caption(f"Arquivo selecionado: **{f.name}**")

        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            back_btn = st.button("⬅️ Voltar para Etapa 2", use_container_width=True)
        with col2:
            extract_btn = st.button("🔎 Extrair deste PDF", use_container_width=True, disabled=(f is None))
        with col3:
            st.caption(f"Modelo atual: **{modelo}**")

        if back_btn:
            st.session_state.stage = "step2"
            st.rerun()

        if extract_btn and f is not None:
            with st.spinner("Extraindo texto do PDF..."):
                text = extract_text_from_pdf(f.getvalue())

            with st.spinner(f"Gerando respostas com modelo **{modelo}**..."):
                answers = run_extraction(modelo, text, st.session_state.rqs, st.session_state.examples)

            # Monta linha de resultado
            import time
            row = {"filename": f.name, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            for i, rq in enumerate(st.session_state.rqs, start=1):
                row[f"RQ{i}"] = answers[i-1] if i-1 < len(answers) else ""

            st.session_state.extracted_rows.append(row)
            st.markdown("### 🧠 Extraction result for this PDF")

            # Tabela por RQ da extração atual
            res_df = pd.DataFrame({
                "RQ": [f"RQ{i+1}: {rq}" for i, rq in enumerate(st.session_state.rqs)],
                "Answer": [answers[i] if i < len(answers) else "" for i in range(len(st.session_state.rqs))]
            })
            st.dataframe(res_df, use_container_width=True)

            # JSON desta extração (útil para copiar/baixar)
            single_result = {
                "filename": f.name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "answers": {f"RQ{i+1}": (answers[i] if i < len(answers) else "") for i in range(len(st.session_state.rqs))}
            }

            # Botão para baixar somente este resultado em JSON
            st.download_button(
                label="📥 Download this result (JSON)",
                data=json.dumps(single_result, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"extraction_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )

            # (Opcional) Mostrar um trecho do texto do artigo para referência rápida
            with st.expander("📝 Show a short snippet of the article text"):
                snippet = text[:1200]
                st.text(snippet)
            st.success("✅ Extração concluída e adicionada à tabela!")

            # após extrair, oferecer botão para extrair outro PDF (reseta o uploader)
            c_again1, c_again2 = st.columns([1,3])
            with c_again1:
                if st.button("➕ Extrair outro PDF", use_container_width=True):
                    st.session_state.upload_counter += 1  # muda a key do uploader
                    st.rerun()
            with c_again2:
                st.caption("Clique para limpar a seleção anterior e enviar o próximo arquivo.")

    # Tabela acumulada
    if st.session_state.extracted_rows:
        st.subheader("📊 Resultados acumulados")
        import pandas as pd
        df = pd.DataFrame(st.session_state.extracted_rows)
        st.dataframe(df, use_container_width=True)

        # Download CSV
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Baixar CSV",
            data=csv_bytes,
            file_name="extracoes.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Limpar tabela
        if st.button("🧹 Limpar resultados", type="secondary"):
            st.session_state.extracted_rows = []
            st.rerun()
