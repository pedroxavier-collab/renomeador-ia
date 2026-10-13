"""
RenomIAdor - Renomeador de arquivos no padrão Expanzio+ com IA
-----------------------------------------------------------------------------
Como funciona:
  1. A pessoa informa o CLIENTE.
  2. Sobe os arquivos.
  3. A IA lê cada arquivo e sugere o NOME ABREVIADO e, se houver, as datas
     de emissão e vencimento.
  4. A pessoa revisa e corrige as sugestões numa tabela editável.
  5. Depois de conferir, baixa tudo renomeado em um .zip.

Padrão final: CLIENTE-NOME ABREVIADO.ext
              CLIENTE-NOME ABREVIADO-EMISSÃO-VENCIMENTO.ext (quando o documento
              tem as duas datas)
Exemplos:     DIA-NOT. EXIG. PUBLI. 29-23.pdf
              DIA-LIC. FUNC-10.01.2026-10.01.2027.pdf

Configuração (Secrets do Streamlit ou variáveis de ambiente):
  IA_PROVEDOR        -> "claude" (padrão) ou "gemini"
  ANTHROPIC_API_KEY  -> chave da API do Claude (se IA_PROVEDOR = claude)
  CLAUDE_MODEL       -> modelo do Claude (opcional)
  GEMINI_API_KEY     -> chave da API do Gemini (se IA_PROVEDOR = gemini)
  GEMINI_MODEL       -> modelo do Gemini (opcional)
"""

import base64
import io
import os
import re
import time
import json
import unicodedata
import zipfile
from datetime import date, datetime
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st
from google import genai
from google.genai import errors as gemini_errors
from google.genai import types as gemini_types


# =========================================================================
# CONFIGURAÇÕES GERAIS
# =========================================================================

PASTA_APP = Path(__file__).parent
CAMINHO_LOGO = PASTA_APP / "assets" / "assinatura-branca-mais-laranja.png"
CAMINHO_ICONE = PASTA_APP / "assets" / "simbolo-branco-mais-laranja.png"

# Provedores de IA disponíveis. Qual usar é definido por IA_PROVEDOR nos
# Secrets; chave e modelo de cada um também ficam nos Secrets.
PROVEDORES = {
    "claude": {
        "nome": "Claude (Anthropic)",
        "secret_chave": "ANTHROPIC_API_KEY",
        "secret_modelo": "CLAUDE_MODEL",
        # Haiku: o mais rápido e barato. Se errar muito na leitura,
        # teste "claude-sonnet-5-5" cadastrando CLAUDE_MODEL nos Secrets.
        "modelo_padrao": "claude-haiku-5-5",
        "pausa": 1,   # segundos entre arquivos
    },
    "gemini": {
        "nome": "Gemini (Google)",
        "secret_chave": "GEMINI_API_KEY",
        "secret_modelo": "GEMINI_MODEL",
        "modelo_padrao": "gemini-2.5-flash",
        "pausa": 4,   # plano gratuito: ~15 requisições por minuto
    },
}
PROVEDOR_PADRAO = "claude"

# Tipos de arquivo aceitos e o "tipo MIME" de cada um
MIME_POR_EXTENSAO = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
TIPOS_ACEITOS = [ext.lstrip(".") for ext in MIME_POR_EXTENSAO]

# Tamanho máximo do NOME ABREVIADO (sem cliente e datas)
TAMANHO_MAX_NOME = 80

# Formato das datas no nome do arquivo. "%d.%m.%Y" → 10.01.2026
# (barra "/" não é permitida em nome de arquivo e o hífen já separa as partes)
FORMATO_DATA_NOME = "%d.%m.%Y"

# Claude: a requisição inteira pode ter até 32 MB, e o arquivo cresce cerca
# de 1/3 ao ser codificado. Por isso o limite do arquivo fica em 20 MB.
LIMITE_CLAUDE_MB = 20
# Gemini: arquivos até este tamanho vão direto na chamada; acima disso,
# upload temporário (apagado logo depois).
LIMITE_GEMINI_DIRETO_MB = 15

# Quantas vezes tentar de novo quando a API responde "limite atingido" (429)
# ou "sobrecarregada" (529)
MAX_TENTATIVAS = 3
PAUSA_APOS_LIMITE = 30

PROMPT = (
    "Analise o conteúdo deste arquivo e extraia três informações, seguindo o "
    "padrão de nomenclatura da Expanzio (regularização imobiliária).\n\n"
    "1) NOME ABREVIADO: abreviação curta e clara que identifique o documento.\n\n"
    "EXEMPLOS DE NOMES ABREVIADOS CORRETOS:\n"
    "- Ata de Reunião → ATA\n"
    "- Cronograma de Legalização → CRONOGRAMA\n"
    "- ART de Execução de Obras do Eng. Marcos David → ART. EXEC. OBRA MARCOS\n"
    "- Certidão de Ônus → CERT. ONUS\n"
    "- Licença de Funcionamento → LIC. FUNC\n"
    "- RRT de Execução de Obras do Bruno Caetano → RRT EXE. OBRA BRUNO\n"
    "- RRT para Elaboração de Projetos do Bruno Caetano → RRT PROJ. BRUNO\n"
    "- Notificação de Exigências Nº 169/2022 (Canteiro de Obras) → "
    "NOT. EXIG. CANTEIRO 169-22\n"
    "- Notificação de Exigências Nº 29/2023 (Engenho Publicitário) → "
    "NOT. EXIG. PUBLI. 29-23\n"
    "- Notificação de Exigências Nº 794/2023 (Estudo Prévio) → "
    "NOT. EXIG. EP. 794-23\n"
    "- Notificação de Exigências Nº 122071912/2023 (Alvará de Construção) → "
    "NOT. EXIG. ALVARA 122071912-23\n"
    "- Certidão Negativa de Débitos do DF Legal → CND DF LEGAL\n\n"
    "REGRAS DO NOME ABREVIADO:\n"
    "- Use TODAS em MAIÚSCULAS (sem acentos).\n"
    "- Abrevie ao máximo MAS mantenha a identificação clara.\n"
    "- Pode usar PONTO após abreviações (ex: LIC., CERT., EXEC.).\n"
    "- Pode usar ESPAÇOS entre palavras.\n"
    "- Inclua números relevantes, como o número do processo ou da notificação.\n"
    "- Para números com ano use NUMERO-AA (ex: 29-23 = nº 29 do ano 2023).\n"
    "- Copie números de processo e notificação exatamente como estão no documento.\n"
    "- NÃO coloque as datas de emissão e vencimento no nome abreviado: "
    "elas vão nos campos próprios abaixo.\n"
    "- NÃO inclua nome do cliente nem extensão do arquivo.\n"
    "- Máximo de 50 caracteres.\n"
    "- Se não conseguir identificar o documento, use: INDEFINIDO\n\n"
    "2) DATA DE EMISSÃO: data em que o documento foi emitido ou expedido.\n"
    "3) DATA DE VENCIMENTO: data de validade ou vencimento do documento.\n\n"
    "REGRAS DAS DATAS:\n"
    "- Só preencha se a data estiver escrita no documento. Não calcule datas "
    "a partir de prazos (ex: 'válido por 90 dias').\n"
    "- Formato AAAA-MM-DD.\n"
    "- Se não houver a data no documento, use null.\n\n"
    "RESPONDA APENAS COM UM JSON NESTE FORMATO, sem markdown e sem texto adicional:\n"
    '{"nome_abreviado": "...", "data_emissao": "AAAA-MM-DD" ou null, '
    '"data_vencimento": "AAAA-MM-DD" ou null}'
)


# =========================================================================
# FUNÇÕES DE TEXTO E NOMENCLATURA
# =========================================================================

def remover_acentos(texto: str) -> str:
    """á → a, ç → c, etc."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def formatar_cliente(texto: str) -> str:
    """Cliente em MAIÚSCULAS, sem acento, só letras, números e espaços."""
    if not texto:
        return ""
    limpo = re.sub(r"[^a-zA-Z0-9 ]", "", remover_acentos(texto)).upper()
    return re.sub(r" +", " ", limpo).strip()


def higienizar_nome_abreviado(nome_bruto: str) -> str:
    """
    Limpa o NOME ABREVIADO (vindo da IA ou digitado na revisão) para que
    vire um nome de arquivo válido. Devolve "" se não sobrar nada.
    """
    if not nome_bruto:
        return ""

    # Só a primeira linha, sem marcações de código e aspas
    nome = str(nome_bruto).strip().split("\n")[0].strip()
    nome = nome.replace("`", "").replace('"', "").replace("'", "")

    nome = remover_acentos(nome).upper()

    # Barra vira hífen (ex: 169/22 → 169-22) em vez de sumir
    nome = nome.replace("/", "-").replace("\\", "-")

    # Mantém apenas letras, números, espaço, ponto, hífen e underline
    nome = re.sub(r"[^A-Z0-9 .\-_]", "", nome)
    nome = re.sub(r" +", " ", nome)

    # Remove pontos, espaços e hífens soltos no começo e no fim
    nome = nome.strip(" .-_")

    return nome[:TAMANHO_MAX_NOME]


def ler_data(valor) -> date | None:
    """Converte "AAAA-MM-DD" (ou um date) em date. Devolve None se inválido."""
    if valor is None:
        return None
    if not isinstance(valor, str) and pd.isna(valor):   # célula vazia (NaT/NaN)
        return None
    if isinstance(valor, datetime):
        return valor.date()
    if isinstance(valor, date):
        return valor
    try:
        return datetime.strptime(str(valor).strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def interpretar_resposta(texto: str) -> dict:
    """
    Lê o JSON devolvido pela IA. Se a IA não respeitar o formato, usa o
    texto inteiro como nome abreviado e deixa as datas em branco.
    """
    texto = (texto or "").strip()
    trecho = re.search(r"\{.*\}", texto, re.DOTALL)
    try:
        dados = json.loads(trecho.group(0)) if trecho else {}
    except json.JSONDecodeError:
        dados = {}

    if not dados:
        return {"nome": higienizar_nome_abreviado(texto),
                "emissao": None, "vencimento": None}

    return {
        "nome": higienizar_nome_abreviado(dados.get("nome_abreviado") or ""),
        "emissao": ler_data(dados.get("data_emissao")),
        "vencimento": ler_data(dados.get("data_vencimento")),
    }


def montar_nome_final(cliente: str, nome_abreviado: str, emissao: date | None,
                      vencimento: date | None, extensao: str) -> str:
    """
    CLIENTE-NOME ABREVIADO.ext
    ou, quando há emissão e vencimento:
    CLIENTE-NOME ABREVIADO-EMISSÃO-VENCIMENTO.ext
    """
    nome = f"{cliente}-{nome_abreviado}"
    if emissao and vencimento:
        nome += (f"-{emissao.strftime(FORMATO_DATA_NOME)}"
                 f"-{vencimento.strftime(FORMATO_DATA_NOME)}")
    return nome + extensao


# =========================================================================
# FUNÇÕES DE IA
# =========================================================================

class ArquivoGrandeDemais(Exception):
    """Arquivo maior do que a IA aceita numa chamada."""


def codigo_http(erro: Exception):
    """Código de status HTTP do erro (429, 401...), se houver."""
    if isinstance(erro, anthropic.APIStatusError):
        return erro.status_code
    if isinstance(erro, gemini_errors.APIError):
        return erro.code
    return None


def vale_tentar_de_novo(erro: Exception) -> bool:
    """Limite de requisições (429) ou serviço sobrecarregado (529)."""
    return codigo_http(erro) in (429, 529) or "RESOURCE_EXHAUSTED" in str(erro)


def mensagem_amigavel(erro: Exception, modelo: str, secret_modelo: str) -> str:
    """Traduz os erros mais comuns para algo que a equipe entenda."""
    codigo = codigo_http(erro)
    texto = str(erro).lower()

    if isinstance(erro, ArquivoGrandeDemais):
        return str(erro)
    if codigo == 429 or "resource_exhausted" in texto:
        return ("Limite de requisições da API atingido mesmo após novas "
                "tentativas. Espere 1 minuto e gere de novo.")
    if codigo == 529:
        return "Serviço de IA sobrecarregado. Tente de novo em alguns minutos."
    if "credit balance" in texto:
        return "Os créditos da API acabaram. Fale com o responsável pelo app."
    if codigo in (401, 403) or "api key not valid" in texto:
        return "Chave da API inválida ou sem permissão."
    if codigo == 404:
        return (f"O modelo '{modelo}' não está disponível para esta chave. "
                f"Ajuste {secret_modelo} nos Secrets do Streamlit.")
    if codigo == 413:
        return "Arquivo grande demais para a IA. Preencha o nome manualmente."
    if codigo == 400 and "page" in texto:
        return "PDF com páginas demais para a IA. Preencha o nome manualmente."
    return str(erro)


def perguntar_ao_claude(cliente_ia: anthropic.Anthropic, modelo: str,
                        conteudo: bytes, extensao: str) -> str:
    """Envia o arquivo ao Claude e devolve a resposta em texto."""
    if len(conteudo) > LIMITE_CLAUDE_MB * 1024 * 1024:
        raise ArquivoGrandeDemais(
            f"Arquivo acima de {LIMITE_CLAUDE_MB} MB. Preencha o nome manualmente."
        )

    mime = MIME_POR_EXTENSAO[extensao]
    dados_b64 = base64.standard_b64encode(conteudo).decode("utf-8")

    if extensao == ".pdf":
        bloco_arquivo = {"type": "document",
                         "source": {"type": "base64", "media_type": mime,
                                    "data": dados_b64}}
    elif extensao == ".txt":
        texto = conteudo.decode("utf-8", errors="replace")[:100_000]
        bloco_arquivo = {"type": "text",
                         "text": f"CONTEÚDO DO ARQUIVO:\n\n{texto}"}
    else:  # imagens
        bloco_arquivo = {"type": "image",
                         "source": {"type": "base64", "media_type": mime,
                                    "data": dados_b64}}

    resposta = cliente_ia.messages.create(
        model=modelo,
        max_tokens=300,
        messages=[{
            "role": "user",
            # O arquivo vem antes da pergunta, como recomenda a documentação
            "content": [bloco_arquivo, {"type": "text", "text": PROMPT}],
        }],
    )
    return "".join(b.text for b in resposta.content if b.type == "text")


def perguntar_ao_gemini(cliente_ia: genai.Client, modelo: str,
                        conteudo: bytes, extensao: str) -> str:
    """Envia o arquivo ao Gemini e devolve a resposta em texto."""
    mime = MIME_POR_EXTENSAO[extensao]
    arquivo_remoto = None
    try:
        if len(conteudo) <= LIMITE_GEMINI_DIRETO_MB * 1024 * 1024:
            parte_arquivo = gemini_types.Part.from_bytes(data=conteudo, mime_type=mime)
        else:
            arquivo_remoto = cliente_ia.files.upload(
                file=io.BytesIO(conteudo),
                config=gemini_types.UploadFileConfig(mime_type=mime),
            )
            parte_arquivo = arquivo_remoto

        resposta = cliente_ia.models.generate_content(
            model=modelo,
            contents=[parte_arquivo, PROMPT],
            config=gemini_types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        return resposta.text or ""
    finally:
        if arquivo_remoto is not None:
            try:
                cliente_ia.files.delete(name=arquivo_remoto.name)
            except Exception:
                pass


def criar_cliente_ia(provedor: str, api_key: str):
    if provedor == "claude":
        # max_retries=0: as novas tentativas são feitas por este app,
        # que mostra na tela quanto tempo está esperando
        return anthropic.Anthropic(api_key=api_key, max_retries=0)
    return genai.Client(api_key=api_key)


def analisar_documento(provedor: str, cliente_ia, modelo: str,
                       conteudo: bytes, nome_arquivo: str) -> dict:
    """
    Pergunta à IA escolhida e devolve:
    {"nome": NOME ABREVIADO, "emissao": date ou None, "vencimento": date ou None}
    """
    extensao = os.path.splitext(nome_arquivo)[1].lower()
    perguntar = perguntar_ao_claude if provedor == "claude" else perguntar_ao_gemini
    return interpretar_resposta(perguntar(cliente_ia, modelo, conteudo, extensao))


def gerar_com_novas_tentativas(provedor, cliente_ia, modelo, conteudo,
                               nome_arquivo, avisar=None) -> dict:
    """Chama a IA e, se bater no limite ou sobrecarga, espera e tenta de novo."""
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            return analisar_documento(provedor, cliente_ia, modelo,
                                      conteudo, nome_arquivo)
        except Exception as erro:
            if not vale_tentar_de_novo(erro) or tentativa == MAX_TENTATIVAS:
                raise
            if avisar:
                avisar(f"API ocupada. Aguardando {PAUSA_APOS_LIMITE}s "
                       f"(tentativa {tentativa + 1} de {MAX_TENTATIVAS})...")
            time.sleep(PAUSA_APOS_LIMITE)


# =========================================================================
# OUTRAS FUNÇÕES
# =========================================================================

def ler_configuracao(nome: str, padrao: str = "") -> str:
    """Lê um valor dos Secrets do Streamlit ou, se não houver, do ambiente."""
    try:
        valor = st.secrets.get(nome, "")
        if valor:
            return valor
    except Exception:
        pass  # sem arquivo de secrets (rodando local)
    return os.environ.get(nome, padrao)


def montar_zip(arquivos: list[tuple[str, bytes]]) -> bytes:
    """Monta o .zip direto na memória. Nada fica gravado no servidor."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for nome, conteudo in arquivos:
            zf.writestr(nome, conteudo)
    return buffer.getvalue()


# =========================================================================
# INTERFACE
# =========================================================================

st.set_page_config(
    page_title="RenomIAdor · Expanzio+",
    page_icon=str(CAMINHO_ICONE) if CAMINHO_ICONE.exists() else None,
    layout="centered",
)

# Paleta oficial Expanzio+
COR_LARANJA = "#FB4710"
COR_BRANCO = "#FFFFFF"
COR_CINZA = "#666666"

# Fontes (Lato e Open Sans) são carregadas pelo .streamlit/config.toml.
# Aqui só deixamos os títulos na versão condensada, como pede a marca.
st.markdown(
    f"""
<style>
h1, h2, h3 {{ font-stretch: 75%; }}

/* Esconde menu e rodapé padrão (o header fica, por causa do botão da sidebar) */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background-color: transparent; }}

.main .block-container {{ padding-top: 2rem; max-width: 900px; }}

h1 {{
    font-weight: 800 !important;
    letter-spacing: -0.01em;
    text-transform: uppercase;
}}
h3 {{ margin-top: 1.5rem; }}

.stButton > button[kind="primary"],
.stDownloadButton > button {{
    background-color: {COR_LARANJA};
    color: {COR_BRANCO};
    border: none;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button:hover {{ background-color: #E03D0A; }}

[data-testid="stFileUploader"] section {{
    border: 2px dashed {COR_LARANJA};
}}
.stProgress > div > div > div > div {{ background-color: {COR_LARANJA}; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: {COR_CINZA} !important; }}

[data-testid="stSidebar"] {{
    background-color: #0A0A0A;
    border-right: 1px solid #1F1F1F;
}}
</style>
""",
    unsafe_allow_html=True,
)

# Memória da sessão: guarda os resultados entre um clique e outro.
# (O Streamlit roda o script do zero a cada interação; sem isso, a tabela
# sumiria ao clicar em "Baixar".)
if "resultados" not in st.session_state:
    st.session_state.resultados = []      # um item por arquivo processado
    st.session_state.assinatura = None    # quais arquivos geraram os resultados
    st.session_state.geracao = 0          # muda a cada nova geração

provedor = ler_configuracao("IA_PROVEDOR", PROVEDOR_PADRAO).strip().lower()
if provedor not in PROVEDORES:
    provedor = PROVEDOR_PADRAO
config_ia = PROVEDORES[provedor]
api_key = ler_configuracao(config_ia["secret_chave"])
modelo = ler_configuracao(config_ia["secret_modelo"], config_ia["modelo_padrao"])

# ---- Barra lateral ------------------------------------------------------
with st.sidebar:
    if CAMINHO_LOGO.exists():
        st.image(str(CAMINHO_LOGO), width=180)

    st.markdown("### Status")
    if api_key:
        st.success("Chave da API configurada")
    else:
        st.error("Chave da API não configurada. Fale com o responsável pelo app.")

    st.markdown("---")
    st.markdown("**Padrão Expanzio:**")
    st.code("CLIENTE-NOME ABREVIADO\nCLIENTE-NOME ABREVIADO-EMISSÃO-VENCIMENTO",
            language=None)
    st.caption("Ex: DIA-NOT. EXIG. PUBLI. 29-23  \n"
               "DIA-LIC. FUNC-10.01.2026-10.01.2027  \n"
               "As datas entram quando o documento tem emissão e vencimento.")

    st.markdown("---")
    st.markdown("**Formatos suportados:** PDF, TXT, PNG, JPG")
    st.caption(f"IA: {config_ia['nome']}  \nModelo: `{modelo}`")

st.title("Renomeador Inteligente de Arquivos")
st.caption("A IA sugere os nomes, você confere e baixa os arquivos renomeados.")

# =========================================================================
# ETAPA 1 - PADRÃO
# =========================================================================
st.markdown("### 1. Informe o cliente")

cliente_digitado = st.text_input(
    "Cliente",
    placeholder="Ex: DIA, BK, SMART, GUSTAVO NINOMIA, TOLLSTADIUS",
    help=("Abreviação do cliente, conforme padrão Expanzio. "
          "Corporativos: sigla (DIA, BK, SMART). "
          "Residenciais: nome + sobrenome ou só o sobrenome marcante."),
)
cliente = formatar_cliente(cliente_digitado)
if cliente_digitado and cliente != cliente_digitado.strip().upper():
    st.caption(f"Cliente formatado: `{cliente}`")


# =========================================================================
# ETAPA 2 - ARQUIVOS
# =========================================================================
st.markdown("### 2. Selecione os arquivos")

arquivos_enviados = st.file_uploader(
    "Arraste os arquivos aqui ou clique para selecionar",
    type=TIPOS_ACEITOS,
    accept_multiple_files=True,
    label_visibility="collapsed",
)

# Ordem alfabética, para a tabela de revisão ficar organizada
arquivos = sorted(arquivos_enviados or [], key=lambda a: a.name.lower())
if arquivos:
    st.caption(f"{len(arquivos)} arquivo(s) selecionado(s).")

# Se a lista de arquivos mudou, os resultados anteriores não valem mais
assinatura_atual = tuple((a.name, a.size) for a in arquivos)
if st.session_state.assinatura != assinatura_atual:
    st.session_state.resultados = []
    st.session_state.assinatura = assinatura_atual

# =========================================================================
# ETAPA 3 - SUGESTÕES DA IA
# =========================================================================
st.markdown("### 3. Gere as sugestões de nome")

pendencias = []
if not api_key:
    pendencias.append("a chave da API não está configurada")
if not cliente:
    pendencias.append("informe o cliente na etapa 1")
if not arquivos:
    pendencias.append("selecione os arquivos na etapa 2")

if pendencias:
    st.info("Para continuar: " + "; ".join(pendencias) + ".")

gerar = st.button(
    "Gerar sugestões de nome",
    type="primary",
    disabled=bool(pendencias),
    width="stretch",
)

if gerar:
    cliente_ia = criar_cliente_ia(provedor, api_key)
    barra = st.progress(0.0, text="Iniciando...")
    aviso = st.empty()
    resultados = []
    total = len(arquivos)

    for posicao, arquivo in enumerate(arquivos, start=1):
        barra.progress((posicao - 1) / total,
                       text=f"Lendo {posicao}/{total}: {arquivo.name}")
        conteudo = arquivo.getvalue()
        analise = {"nome": "", "emissao": None, "vencimento": None}
        erro_texto = ""

        try:
            analise = gerar_com_novas_tentativas(
                provedor, cliente_ia, modelo, conteudo, arquivo.name,
                avisar=aviso.info,
            )
            if analise["nome"] in ("", "INDEFINIDO"):
                erro_texto = "A IA não identificou o documento. Preencha manualmente."
                analise["nome"] = ""
        except Exception as erro:
            erro_texto = "Erro: " + mensagem_amigavel(
                erro, modelo, config_ia["secret_modelo"]
            )

        aviso.empty()
        resultados.append({
            "original": arquivo.name,
            "extensao": os.path.splitext(arquivo.name)[1].lower(),
            "conteudo": conteudo,
            "sugestao": analise["nome"],
            "emissao": analise["emissao"],
            "vencimento": analise["vencimento"],
            "observacao": erro_texto,
        })

        if posicao < total:
            time.sleep(config_ia["pausa"])

    barra.empty()
    st.session_state.resultados = resultados
    st.session_state.geracao += 1

# =========================================================================
# ETAPA 4 - REVISÃO E DOWNLOAD
# =========================================================================
resultados = st.session_state.resultados

if resultados:
    st.markdown("### 4. Revise os nomes")
    st.caption(
        "Confira cada sugestão com o arquivo original. Clique na célula para "
        "corrigir o nome abreviado ou as datas. Arquivos que a IA não "
        "conseguiu ler aparecem em branco para preencher manualmente. "
        "As datas só entram no nome quando emissão e vencimento estão preenchidas."
    )

    falhas = sum(1 for r in resultados if r["observacao"])
    com_datas = sum(1 for r in resultados if r["emissao"] and r["vencimento"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Arquivos", len(resultados))
    c2.metric("Com emissão e vencimento", com_datas)
    c3.metric("Preencher manualmente", falhas)

    tabela_base = pd.DataFrame([
        {
            "Arquivo original": r["original"],
            "Nome abreviado": r["sugestao"],
            "Emissão": r["emissao"],
            "Vencimento": r["vencimento"],
            "Observação": r["observacao"],
        }
        for r in resultados
    ])
    # Garante que as colunas de data sejam tratadas como data mesmo vazias
    for coluna in ("Emissão", "Vencimento"):
        tabela_base[coluna] = pd.to_datetime(tabela_base[coluna]).dt.date

    tabela_editada = st.data_editor(
        tabela_base,
        key=f"editor_{st.session_state.geracao}",
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        disabled=["Arquivo original", "Observação"],
        column_config={
            "Arquivo original": st.column_config.TextColumn(width="medium"),
            "Nome abreviado": st.column_config.TextColumn(
                width="large", max_chars=TAMANHO_MAX_NOME,
            ),
            "Emissão": st.column_config.DateColumn(format="DD/MM/YYYY", width="small"),
            "Vencimento": st.column_config.DateColumn(format="DD/MM/YYYY", width="small"),
            "Observação": st.column_config.TextColumn(width="medium"),
        },
    )

    # ---- Monta os nomes finais e confere problemas ----------------------
    problemas = []
    so_uma_data = []
    linhas_finais = []
    nomes_usados = {}

    for item, (_, linha) in zip(resultados, tabela_editada.iterrows()):
        nome_abreviado = higienizar_nome_abreviado(linha["Nome abreviado"])
        emissao = ler_data(linha["Emissão"])
        vencimento = ler_data(linha["Vencimento"])

        if not nome_abreviado:
            problemas.append(f"{item['original']}: nome abreviado em branco.")
        if emissao and vencimento and vencimento < emissao:
            problemas.append(f"{item['original']}: vencimento anterior à emissão.")
        if bool(emissao) != bool(vencimento):
            so_uma_data.append(item["original"])

        nome_final = montar_nome_final(
            cliente, nome_abreviado, emissao, vencimento, item["extensao"]
        ) if nome_abreviado else ""

        # Sem numeração, dois arquivos podem acabar com o mesmo nome
        if nome_final:
            chave_nome = nome_final.upper()
            if chave_nome in nomes_usados:
                problemas.append(
                    f"{nomes_usados[chave_nome]} e {item['original']} ficariam com "
                    f"o mesmo nome. Diferencie o nome abreviado de um deles."
                )
            nomes_usados[chave_nome] = item["original"]

        linhas_finais.append({
            "original": item["original"],
            "nome_final": nome_final,
            "conteudo": item["conteudo"],
        })

    if not cliente:
        problemas.append("Informe o cliente na etapa 1.")

    st.markdown("**Como os arquivos vão ficar:**")
    st.dataframe(
        pd.DataFrame(
            [{"Arquivo original": l["original"], "Nome final": l["nome_final"]}
             for l in linhas_finais]
        ),
        hide_index=True,
        width="stretch",
    )

    if so_uma_data:
        st.info("Só uma das datas está preenchida em: " + ", ".join(so_uma_data)
                + ". Nesses arquivos as datas não entram no nome.")

    if problemas:
        st.warning("Corrija antes de baixar:\n\n" +
                   "\n".join(f"- {p}" for p in problemas))

    conferido = st.checkbox(
        "Conferi todos os nomes acima e estão corretos",
        key=f"conferido_{st.session_state.geracao}",
        disabled=bool(problemas),
    )

    pode_baixar = conferido and not problemas
    zip_bytes = (
        montar_zip([(l["nome_final"], l["conteudo"]) for l in linhas_finais])
        if pode_baixar else b""
    )

    st.download_button(
        "Baixar arquivos renomeados (.zip)",
        data=zip_bytes,
        file_name=f"{cliente or 'arquivos'}-renomeados.zip",
        mime="application/zip",
        type="primary",
        disabled=not pode_baixar,
        on_click="ignore",  # baixar não recarrega a página
        width="stretch",
    )
