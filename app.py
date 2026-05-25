"""
Renomeador Inteligente de Arquivos com IA (Google Gemini)
----------------------------------------------------------
Aplicação Streamlit que recebe múltiplos arquivos, envia para a API do Gemini
para identificar o conteúdo, renomeia automaticamente e devolve tudo em um .zip.

Autor: Engenheiro Sênior (exemplo didático)
"""

import io
import os
import re
import time
import tempfile
import zipfile

import streamlit as st
import google.generativeai as genai


# =========================================================================
# CONFIGURAÇÕES GLOBAIS
# =========================================================================

# Modelo do Gemini que será usado. "flash" é rápido e barato — ideal para
# este caso de uso. Você pode trocar por "gemini-2.0-flash" se preferir.
MODEL_NAME = "gemini-1.5-flash"

# Tipos de arquivo aceitos no upload
TIPOS_ACEITOS = ["pdf", "txt", "png", "jpg", "jpeg"]

# Tamanho máximo do nome de arquivo (sem extensão) para evitar nomes gigantes
TAMANHO_MAX_NOME = 80


# =========================================================================
# FUNÇÕES AUXILIARES (a lógica do app fica separada da UI)
# =========================================================================

def higienizar_nome(nome_bruto: str, extensao_original: str) -> str:
    """
    Limpa a resposta do Gemini para garantir que vire um nome de arquivo válido.

    - Remove markdown (```, **, etc.)
    - Remove aspas, espaços, barras, dois-pontos e outros caracteres proibidos
    - Garante que a extensão original seja preservada
    """
    # Pega só a primeira linha (caso o Gemini retorne texto extra)
    nome = nome_bruto.strip().split("\n")[0].strip()

    # Remove blocos de código markdown e aspas
    nome = nome.replace("```", "").replace("`", "")
    nome = nome.replace('"', "").replace("'", "")

    # Caracteres proibidos em sistemas de arquivos (Windows + Linux + Mac)
    nome = re.sub(r'[<>:"/\\|?*\n\r\t]', "", nome)

    # Troca espaços por underline
    nome = re.sub(r"\s+", "_", nome)

    # Mantém apenas letras, números, underline, hífen e ponto
    nome = re.sub(r"[^a-zA-Z0-9_\-\.]", "", nome)

    # Remove pontos e underlines do começo e do fim
    nome = nome.strip("._-")

    # Se o Gemini já incluiu a extensão, remove para tratar separadamente
    base, ext_resposta = os.path.splitext(nome)
    if not base:
        base = "arquivo_sem_nome"

    # Limita o tamanho do nome
    base = base[:TAMANHO_MAX_NOME]

    # Sempre força a extensão original (a do arquivo que o usuário enviou)
    return f"{base}{extensao_original.lower()}"


def gerar_nome_via_gemini(conteudo_bytes: bytes,
                          nome_original: str,
                          api_key: str) -> str:
    """
    Envia o arquivo para a API do Gemini e devolve o novo nome sugerido.

    Lança exceção em caso de erro (rate limit, arquivo inválido, etc.) — quem
    chama essa função decide como tratar.
    """
    genai.configure(api_key=api_key)

    extensao = os.path.splitext(nome_original)[1] or ""

    # O Gemini precisa do arquivo em disco (a SDK faz upload depois).
    # Por isso salvamos em um arquivo temporário.
    arquivo_temp = tempfile.NamedTemporaryFile(delete=False, suffix=extensao)
    arquivo_temp.write(conteudo_bytes)
    arquivo_temp.close()

    arquivo_no_gemini = None
    try:
        # 1) Faz upload do arquivo para os servidores do Gemini
        arquivo_no_gemini = genai.upload_file(path=arquivo_temp.name)

        # 2) Prompt — instruções bem específicas pra IA não viajar
        prompt = (
            "Analise o conteúdo deste arquivo e gere um nome curto, "
            "descritivo e padronizado para ele.\n\n"
            "REGRAS OBRIGATÓRIAS (siga à risca):\n"
            "- Responda APENAS com o nome do arquivo. Nada mais.\n"
            "- NÃO use markdown, aspas, ou qualquer formatação.\n"
            "- NÃO inclua a extensão do arquivo.\n"
            "- Use APENAS letras (sem acento), números, underline (_) "
            "e hífen (-).\n"
            "- NÃO use espaços, barras, dois-pontos ou caracteres especiais.\n"
            "- Máximo de 60 caracteres.\n"
            "- Seja descritivo e útil.\n\n"
            "Exemplos de respostas corretas:\n"
            "Fatura_Energia_Janeiro_2026\n"
            "Contrato_Aluguel_Joao_Silva_2025\n"
            "Foto_Cachorro_Praia\n"
            "Comprovante_Pix_Loja_Tech\n"
        )

        # 3) Chama o modelo passando o prompt + o arquivo
        modelo = genai.GenerativeModel(MODEL_NAME)
        resposta = modelo.generate_content([prompt, arquivo_no_gemini])

        nome_bruto = resposta.text or ""
        return higienizar_nome(nome_bruto, extensao)

    finally:
        # Limpeza: apaga o arquivo temporário local
        try:
            os.unlink(arquivo_temp.name)
        except OSError:
            pass
        # E apaga o arquivo nos servidores do Gemini (boa prática)
        if arquivo_no_gemini is not None:
            try:
                genai.delete_file(arquivo_no_gemini.name)
            except Exception:
                pass


def evitar_nome_duplicado(nome: str, ja_usados: dict) -> str:
    """
    Se dois arquivos receberem o mesmo nome do Gemini, adiciona _2, _3, etc.
    'ja_usados' é um dicionário que vai sendo atualizado a cada chamada.
    """
    if nome not in ja_usados:
        ja_usados[nome] = 1
        return nome

    base, ext = os.path.splitext(nome)
    ja_usados[nome] += 1
    novo = f"{base}_{ja_usados[nome]}{ext}"
    # Garante que o novo nome também é único
    return evitar_nome_duplicado(novo, ja_usados)


def criar_zip_em_memoria(pasta_arquivos: str) -> bytes:
    """Compacta todos os arquivos de uma pasta em um .zip em memória."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for nome_arquivo in os.listdir(pasta_arquivos):
            caminho = os.path.join(pasta_arquivos, nome_arquivo)
            if os.path.isfile(caminho):
                zf.write(caminho, arcname=nome_arquivo)
    buffer.seek(0)
    return buffer.getvalue()


# =========================================================================
# INTERFACE (Streamlit)
# =========================================================================

st.set_page_config(
    page_title="Renomeador IA",
    page_icon="📁",
    layout="centered",
)

st.title("📁 Renomeador Inteligente de Arquivos")
st.caption("Use a IA do Google Gemini para renomear seus arquivos automaticamente.")

# ---- Barra lateral: configuração ---------------------------------------
with st.sidebar:
    st.header("⚙️ Configuração")
    api_key = st.text_input(
        "Chave da API Gemini",
        type="password",
        help="Pegue sua chave gratuita em https://aistudio.google.com/apikey",
    )
    st.markdown("---")
    st.markdown(
        "**Formatos suportados:**\n"
        "- PDF\n- TXT\n- PNG\n- JPG / JPEG"
    )
    st.markdown("---")
    st.caption(f"Modelo em uso: `{MODEL_NAME}`")

# ---- Área principal: upload --------------------------------------------
arquivos_enviados = st.file_uploader(
    "Selecione os arquivos para renomear",
    type=TIPOS_ACEITOS,
    accept_multiple_files=True,
)

if arquivos_enviados:
    st.info(f"📎 {len(arquivos_enviados)} arquivo(s) selecionado(s).")

# Avisos para o usuário antes de processar
botao_desabilitado = not (arquivos_enviados and api_key)
if arquivos_enviados and not api_key:
    st.warning("⚠️ Informe sua chave da API Gemini na barra lateral.")

processar = st.button(
    "🚀 Processar Arquivos",
    type="primary",
    disabled=botao_desabilitado,
)

# ---- Lógica de processamento -------------------------------------------
if processar:
    barra_progresso = st.progress(0.0, text="Iniciando...")
    log = st.container()

    # Pasta temporária para guardar os arquivos renomeados
    pasta_temp = tempfile.mkdtemp(prefix="renomeados_")

    nomes_ja_usados: dict = {}
    resumo = {"sucesso": 0, "erro": 0}
    total = len(arquivos_enviados)

    for indice, arquivo in enumerate(arquivos_enviados, start=1):
        progresso = (indice - 1) / total
        barra_progresso.progress(
            progresso,
            text=f"Processando {indice}/{total}: {arquivo.name}",
        )

        try:
            conteudo = arquivo.getvalue()

            # Chama a IA — esta é a parte que pode falhar
            novo_nome = gerar_nome_via_gemini(
                conteudo_bytes=conteudo,
                nome_original=arquivo.name,
                api_key=api_key,
            )
            novo_nome = evitar_nome_duplicado(novo_nome, nomes_ja_usados)

            # Salva o arquivo com o novo nome na pasta temporária
            caminho_destino = os.path.join(pasta_temp, novo_nome)
            with open(caminho_destino, "wb") as f:
                f.write(conteudo)

            resumo["sucesso"] += 1
            with log:
                st.success(f"✅ `{arquivo.name}` → `{novo_nome}`")

        except Exception as erro:
            # Não interrompe o processamento — apenas avisa e segue
            resumo["erro"] += 1
            mensagem = str(erro)

            # Mensagem mais amigável para erros conhecidos
            if "429" in mensagem or "quota" in mensagem.lower() or "rate" in mensagem.lower():
                texto_amigavel = "Limite de requisições da API atingido. Aguarde alguns segundos."
            elif "401" in mensagem or "API key" in mensagem:
                texto_amigavel = "Chave da API inválida ou sem permissão."
            else:
                texto_amigavel = mensagem

            with log:
                st.error(f"❌ `{arquivo.name}` — {texto_amigavel}")

            # Pequena pausa em caso de rate limit
            if "rate" in mensagem.lower() or "429" in mensagem:
                time.sleep(2)

    # Finaliza a barra
    barra_progresso.progress(1.0, text="Processamento concluído!")

    # Resumo final
    st.markdown("### 📊 Resumo")
    col_a, col_b = st.columns(2)
    col_a.metric("✅ Renomeados", resumo["sucesso"])
    col_b.metric("❌ Falhas", resumo["erro"])

    # Botão de download — só se algum arquivo deu certo
    if resumo["sucesso"] > 0:
        zip_bytes = criar_zip_em_memoria(pasta_temp)
        st.download_button(
            label="📥 Baixar tudo em .zip",
            data=zip_bytes,
            file_name="arquivos_renomeados.zip",
            mime="application/zip",
            type="primary",
        )
    else:
        st.warning("Nenhum arquivo foi processado com sucesso.")
