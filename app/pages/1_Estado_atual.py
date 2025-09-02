import streamlit as st

st.set_page_config(page_title="Estado Atual", layout="wide")

st.title("📌 Estado atual da sessão")

st.markdown("Aqui você pode visualizar tudo que está armazenado no `st.session_state`.")

# Mostra o conteúdo inteiro
st.json(st.session_state)
