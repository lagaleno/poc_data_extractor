# PoC Extrator de PDF para Estudos Secundários

Este projeto é uma prova de conceito para auxiliar pesquisadores a realizarem **extração automática de dados de artigos científicos em PDF**, utilizando diferentes modelos de IA generativa, interface em **Streamlit**, manipulação com **Pandas** e exportação para **Google Planilhas**.

---

## 🚀 Como rodar o projeto na sua máquina

### 1. Pré-requisitos
- [Python 3.11+](https://www.python.org/downloads/) instalado  
- Git Bash (ou outro terminal de sua preferência)

---

### 2. Clonar o repositório
```bash
git clone <URL_DO_REPOSITORIO>
cd poc_extrator_pdf
```

---

### 3. Criar ambiente virtual
```bash
python -m venv venv
```

Ativar o ambiente virtual:

- **Git Bash / Linux / MacOS**
  ```bash
  source venv/Scripts/activate
  ```
- **Prompt de Comando (cmd – Windows)**
  ```cmd
  venv\Scripts\activate
  ```
- **PowerShell (Windows)**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```

Se aparecer `(venv)` no início da linha do terminal, o ambiente foi ativado com sucesso ✅

---

### 4. Instalar dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 5. Rodar a aplicação
```bash
streamlit run app/streamlit_app.py
```

A interface abrirá automaticamente no navegador em:  
[http://localhost:8501](http://localhost:8501)

---

### 6. Configuração opcional (APIs e Google Sheets)
Crie um arquivo `.env` na raiz do projeto (baseado em `.env.example`) e preencha com suas chaves de API:

```env
OPENAI_API_KEY=xxxx
ANTHROPIC_API_KEY=xxxx
GOOGLE_API_KEY=xxxx
GOOGLE_SHEETS_KEY_FILE=path/para/sua-chave.json
```

Isso permitirá usar modelos generativos (OpenAI, Gemini, LLaMA, etc.) e exportar os resultados para o Google Sheets.

---

## 📂 Estrutura de Diretórios

```
poc_extrator_pdf/
├── app/
│   ├── streamlit_app.py        # interface principal
│   ├── extractors/             # extratores de diferentes modelos (OpenAI, Gemini, LLaMA, etc.)
│   ├── utils/                  # funções auxiliares (PDF, Sheets, storage)
│   └── state/                  # gerenciamento de sessão e fluxo
├── config/                     # configurações e schema dos campos
├── data/                       # uploads, exemplos e saídas
├── tests/                      # testes unitários
├── requirements.txt
├── README.md
└── .env.example
```

---

## 🛠️ Roadmap sugerido
1. Interface inicial: escolha do modelo + entrada de objetivo e perguntas.  
2. Coleta de exemplos (artigos + respostas).  
3. Extração automática de 1 artigo por vez.  
4. Exportação para CSV/Google Sheets.  
5. Avaliação da qualidade das extrações.

---
