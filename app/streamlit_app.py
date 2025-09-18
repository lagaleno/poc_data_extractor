# app/streamlit_app.py
# --- ensure project root is on sys.path ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # .../poc_extrator_pdf
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------
import streamlit as st
from app.extractors import make_extractor
from app.utils.pdf_loader import extract_text_from_pdf

# ==============================
# Helpers da Etapa 3 (coloque no topo do arquivo se ainda não existirem)
# ==============================
# ===== EN-only helpers (replace the previous helpers) =====
import io, re, time
import pdfplumber
import pandas as pd
import json

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
    ["OpenAI", "Gemini", "Heurístico (Regex)"],
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
        if st.button("➡️ Ir para a etapa 2 — Exemplos", width='stretch'):
            st.session_state.stage = "step2"
            st.rerun()
    else:
        st.button("➡️ Ir para a etapa 2 — Exemplos", disabled=True, width='stretch')
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
            back_btn = st.button("⬅️ Voltar para Etapa 2", width='stretch')
        with col2:
            extract_btn = st.button("🔎 Extrair deste PDF", width='stretch', disabled=(f is None))
        with col3:
            st.caption(f"Modelo atual: **{modelo}**")

        if back_btn:
            st.session_state.stage = "step2"
            st.rerun()

        if extract_btn and f is not None:
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(f.getvalue())

            with st.spinner(f"Answering RQs with **{modelo}**..."):
                extractor = make_extractor(modelo, rqs=st.session_state.rqs, examples=st.session_state.examples)
                answers = extractor.extract(text)

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
            st.dataframe(res_df, width='stretch')

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
                width='stretch'
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
        st.dataframe(df, width='stretch')

        # Download CSV
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Baixar CSV",
            data=csv_bytes,
            file_name="extracoes.csv",
            mime="text/csv",
            width='stretch'
        )

        # Limpar tabela
        if st.button("🧹 Limpar resultados", type="secondary"):
            st.session_state.extracted_rows = []
            st.rerun()
