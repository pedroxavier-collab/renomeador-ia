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
# este caso de uso. Outras opções: "gemini-2.0-flash" ou "gemini-2.5-pro".
MODEL_NAME = "gemini-2.5-flash"

# Tipos de arquivo aceitos no upload
TIPOS_ACEITOS = ["pdf", "txt", "png", "jpg", "jpeg"]

# Tamanho máximo do nome de arquivo (sem extensão) para evitar nomes gigantes
TAMANHO_MAX_NOME = 80

# Pausa (em segundos) entre o processamento de cada arquivo.
# O plano grátis do Gemini permite ~15 requisições por minuto.
# 4 segundos = 15 req/min, ficando bem dentro do limite.
PAUSA_ENTRE_ARQUIVOS = 4

# Quantas vezes tentar de novo se a API responder rate limit (429)
MAX_TENTATIVAS = 3

# Pausa (em segundos) ao receber rate limit, antes de tentar de novo
PAUSA_APOS_RATE_LIMIT = 30


# =========================================================================
# FUNÇÕES AUXILIARES (a lógica do app fica separada da UI)
# =========================================================================

def montar_nome_padrao_expanzio(cliente: str,
                                numero: int,
                                nome_abreviado: str,
                                revisao: str,
                                extensao: str) -> str:
    """
    Monta o nome final no padrão Expanzio: CLIENTE-000-NOME ABREVIADO-REVISÃO

    Exemplo: DIA-003-NOT. EXIG. PUBLI. 29-23-R00.pdf
    """
    # Numeração com 3 dígitos (1 → 001, 12 → 012)
    numero_str = f"{numero:03d}"

    # Garante que a revisão esteja no formato R00, R01, R02...
    revisao = revisao.strip().upper()
    if not revisao:
        revisao = "R00"
    elif not revisao.startswith("R"):
        revisao = f"R{revisao}"

    # Monta: CLIENTE-000-NOME ABREVIADO-R00.ext
    nome_final = f"{cliente}-{numero_str}-{nome_abreviado}-{revisao}{extensao}"
    return nome_final


def higienizar_nome(nome_bruto: str, extensao_original: str) -> str:
    """
    Limpa a resposta do Gemini garantindo que vire um nome de arquivo válido.

    Padrão Expanzio: permite letras, números, pontos, espaços e hífens.
    Remove apenas caracteres que quebram o sistema de arquivos.
    """
    import unicodedata

    # Pega só a primeira linha (caso o Gemini retorne texto extra)
    nome = nome_bruto.strip().split("\n")[0].strip()

    # Remove blocos de código markdown e aspas
    nome = nome.replace("```", "").replace("`", "")
    nome = nome.replace('"', "").replace("'", "")

    # Remove acentos (á → a, ã → a, ç → c)
    nfkd = unicodedata.normalize("NFKD", nome)
    nome = "".join(c for c in nfkd if not unicodedata.combining(c))

    # Caracteres proibidos em sistemas de arquivos (Windows + Linux + Mac)
    nome = re.sub(r'[<>:"/\\|?*\n\r\t]', "", nome)

    # Colapsa múltiplos espaços em um só
    nome = re.sub(r" +", " ", nome).strip()

    # Mantém apenas letras, números, espaço, ponto, hífen e underline
    nome = re.sub(r"[^a-zA-Z0-9 \.\-_]", "", nome)

    # Remove pontos, espaços e hífens do começo e do fim
    nome = nome.strip(" .-_")

    if not nome:
        nome = "arquivo sem nome"

    # Limita o tamanho
    nome = nome[:TAMANHO_MAX_NOME]

    # Garante a extensão original
    return f"{nome}{extensao_original.lower()}"


def gerar_nome_via_gemini(conteudo_bytes: bytes,
                          nome_original: str,
                          api_key: str) -> str:
    """
    Envia o arquivo para a API do Gemini e devolve apenas o NOME ABREVIADO
    no padrão Expanzio (ex: "NOT. EXIG. PUBLI. 29-23").

    O CLIENTE, número e revisão são adicionados depois pela função
    montar_nome_padrao_expanzio() — assim a IA só se preocupa com a
    identificação correta do tipo de documento.

    Lança exceção em caso de erro (rate limit, arquivo inválido, etc.).
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

        # 2) Prompt — agora a IA gera APENAS o NOME ABREVIADO.
        # CLIENTE, numeração e revisão são definidos pelo usuário e adicionados
        # depois pelo código. Isso deixa o resultado mais consistente.
        prompt = (
            "Analise o conteúdo deste arquivo e gere o NOME ABREVIADO do documento "
            "seguindo o padrão de nomenclatura da Expanzio (regularização imobiliária).\n\n"
            "OBJETIVO: criar uma abreviação curta e clara que identifique o documento, "
            "no estilo dos exemplos abaixo.\n\n"
            "EXEMPLOS DE NOMES ABREVIADOS CORRETOS:\n"
            "- Ata de Reunião → ATA\n"
            "- Cronograma de Legalização → CRONOGRAMA\n"
            "- ART de Execução de Obras do Eng. Marcos David → ART. EXEC. OBRA MARCOS\n"
            "- Certidão de Ônus → CERT. ONUS\n"
            "- Licença de Funcionamento de 2025 → LIC. FUNC. 2025\n"
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
            "REGRAS DE ABREVIAÇÃO:\n"
            "- Use TODAS em MAIÚSCULAS (sem acentos).\n"
            "- Abrevie ao máximo MAS mantenha a identificação clara.\n"
            "- Pode usar PONTO após abreviações (ex: LIC., CERT., EXEC.).\n"
            "- Pode usar ESPAÇOS entre palavras.\n"
            "- Inclua datas/números quando relevantes (ano, número do processo).\n"
            "- Para datas curtas use formato ANO-MES ou só o ANO (ex: 29-23 = nº 29 do "
            "ano 2023; 2025 = ano).\n"
            "- NÃO inclua nome do cliente, NÃO inclua numeração sequencial, "
            "NÃO inclua revisão (R00, R01).\n"
            "- NÃO inclua extensão do arquivo.\n"
            "- Máximo de 50 caracteres.\n\n"
            "RESPONDA APENAS COM O NOME ABREVIADO. Sem markdown, sem aspas, sem "
            "explicações, sem texto adicional. Apenas a abreviação."
        )

        # 3) Chama o modelo passando o prompt + o arquivo
        modelo = genai.GenerativeModel(MODEL_NAME)
        resposta = modelo.generate_content([prompt, arquivo_no_gemini])

        nome_bruto = resposta.text or ""
        # Higieniza usando extensão vazia: queremos apenas o NOME ABREVIADO,
        # sem extensão (que será adicionada depois no montar_nome_padrao_expanzio)
        nome_higienizado = higienizar_nome(nome_bruto, "")
        return nome_higienizado

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


def gerar_nome_com_retry(conteudo_bytes: bytes,
                         nome_original: str,
                         api_key: str,
                         callback_status=None) -> str:
    """
    Envolve a chamada ao Gemini com retry automático em caso de rate limit.
    Se cair em 429, espera e tenta de novo até MAX_TENTATIVAS vezes.

    'callback_status' é uma função opcional pra mostrar mensagens na tela
    durante a espera (assim o usuário sabe que o app não travou).
    """
    ultimo_erro = None

    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            return gerar_nome_via_gemini(conteudo_bytes, nome_original, api_key)
        except Exception as erro:
            ultimo_erro = erro
            msg = str(erro).lower()
            eh_rate_limit = "429" in msg or "rate" in msg or "quota" in msg

            # Se não for rate limit, não adianta tentar de novo — propaga o erro
            if not eh_rate_limit:
                raise

            # Se ainda tem tentativa, espera e tenta de novo
            if tentativa < MAX_TENTATIVAS:
                if callback_status:
                    callback_status(
                        f"⏳ Limite atingido. Aguardando {PAUSA_APOS_RATE_LIMIT}s "
                        f"e tentando de novo (tentativa {tentativa + 1}/{MAX_TENTATIVAS})..."
                    )
                time.sleep(PAUSA_APOS_RATE_LIMIT)

    # Se chegou aqui, esgotou as tentativas
    raise ultimo_erro


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


def carregar_api_key() -> str:
    """
    Tenta carregar a chave da API em ordem de prioridade:
    1. Dos 'Secrets' do Streamlit (produção / Streamlit Cloud)
    2. De variável de ambiente GEMINI_API_KEY (rodando local)
    3. Retorna string vazia se não encontrar (aí pedimos pro usuário)
    """
    # Tenta pegar dos Secrets do Streamlit
    try:
        chave = st.secrets.get("GEMINI_API_KEY", "")
        if chave:
            return chave
    except Exception:
        pass  # st.secrets pode não existir no ambiente local

    # Tenta pegar de variável de ambiente
    return os.environ.get("GEMINI_API_KEY", "")


# =========================================================================
# INTERFACE (Streamlit)
# =========================================================================

st.set_page_config(
    page_title="Renomeador IA · Expanzio+",
    page_icon="📁",
    layout="centered",
)

# =========================================================================
# IDENTIDADE VISUAL — Expanzio+
# =========================================================================

# Logo da marca (embutida em base64 para não depender de arquivo externo)
LOGO_BASE64 = "iVBORw0KGgoAAAANSUhEUgAABkAAAAHqCAYAAABcLIKhAAABCGlDQ1BJQ0MgUHJvZmlsZQAAeJxjYGA8wQAELAYMDLl5JUVB7k4KEZFRCuwPGBiBEAwSk4sLGHADoKpv1yBqL+viUYcLcKakFicD6Q9ArFIEtBxopAiQLZIOYWuA2EkQtg2IXV5SUAJkB4DYRSFBzkB2CpCtkY7ETkJiJxcUgdT3ANk2uTmlyQh3M/Ck5oUGA2kOIJZhKGYIYnBncAL5H6IkfxEDg8VXBgbmCQixpJkMDNtbGRgkbiHEVBYwMPC3MDBsO48QQ4RJQWJRIliIBYiZ0tIYGD4tZ2DgjWRgEL7AwMAVDQsIHG5TALvNnSEfCNMZchhSgSKeDHkMyQx6QJYRgwGDIYMZAKbWPz9HbOBQAADfxUlEQVR42uzdeZgUxfkH8G9VHzOzu1yiiUEUUFAuObwCKMgtqCie0XhgDCJqjBJNNPGXGDUx0SQaDQaieJ8xaryN4oGoIFFUFBVFvKIxnojs7sz0UfX7Y7aanmXx4Jjpmfl+nmeePVh2e6q7q6vqrXoLICIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIi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XmAffvghbrnlFhYIAQB22GEH/eqrr+rJkydH34tPdGjrWeG6bvScqBWmzdy6vy2EiNpG8ckspnyklFHgccKECbj11lsxadIkPk+JqDbqThYBEW2oV155RcyZMwf5fB6+70ezNsut9d4TlmUhDEO4rosDDjgAPXr0qIoG38iRI/WECROiTgCARKwAcV03KnvzMZVK4aabbsITTzzB1R/0pUwg9fTTT8eiRYv0mDFj2EHbyIYNG6Y333zzterLSkiBFae1htYadXV16NevH09sFTMryH7yk5+wMIgq/F4GgAULFuD1119nm5AwduxYfcstt6Br167o2LFj0XUSf97ThslkMmhubgYA9OzZEzfccAOOOuooFiwRVT0GQIhoo/jLX/6CN998E47jJGYD3fjSYADRahClFHr37o0jjjiiKsr+1FNPRSaTifLAZrPZRHQQ4hugm8DMypUrMWvWLN4w9JXMkv5sNovddtsNt99+O0499VR20DaiMWPGoF27dlH9aAYbKiEFFoCizU/Ne9hzzz15YqvAV7UjJk2ahKFDh7I+IKpQpt6eM2cOC4MwduxYffvtt2PQoEFoaGiIngFhGBY9DyqlfZJ0mUwGuVwOUkq0a9cO1113HY477jg+U4moqvEJQkQbxTvvvCP+/ve/A1izyXFSmJUfQGEg3vd9AMDxxx9f8eU+evRovf/++8PzvGhmbCaTSUQQypQzAORyOQDAHXfcgQULFnCmH30lz/OiTlpTUxPq6+tx8cUX4+qrr2YHbSMZNmwYbNsuSpmXlBV8X0c88GGCvrvsskvVrO6jYuYZJ4SAZVk4+eSTWShEFUgphVQqhX//+9948MEH2SascePHj9c33ngj2rdvH01U830/SmXMoMfG75+ZPVRyuRyEEPA8D7NmzcLPf/5ztp+IqGrxaUJEG80555wjXn755SgdSbk5jhMdh2lMA4j2HujatSt++ctfVnRD73e/+120mbTv+1EnIQnln06nEQRBtBLkk08+wV//+lfeKPS1xHM+19fXR9f0AQccgLlz5+ptt92WnbQN0LNnT73jjjtGX5ugaSWllzDHagIgSil07doVu+yyC09wlTIBkDAMceihh4IbuBJVHhN0v+SSS1gYNW7UqFH61ltvxeabbx6lUY7vCRdvm4RhyBRYG6l/HO+r+b4fpS3+7W9/i9NOO42FTERV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9abnZnPAdT18qTK98cYbwmyWGN9gOwksy4oCAGYAbcyYMRg7duxGHUD5+c9/rnv37g2tNWzbTsQApVnKbo7F5Mq977778MgjjzCFSA159dVXxa9+9SsxaNAgceyxx+Kpp55K/DGb9F1CCOTzeQgh4LputBG6Ugo777wz9t9//6odDB07diyCIIjSQdXS8zMMwygNkpQSuVwOUkp069YNffr04QB4lXNdF83NzVG7IggCWJaFpqYmnH766SwgqlgHHHBAVMeZ/PsmCFLK+jWe/sr8/c8++wx//etf2T6sIXvssUeUJtW0sZKuvr4ev//973HIIYck9lqdM2eOOO644/D+++/Dsiw0NzejU6dOiSnfeKrxeJrVbDaLvfbaizfGRtAX0LsAegCg+4vCxwGAHog1n6/vayCgB7d87CegdxSFr3cB9E4tr8ECepCAHgTobQG2m6kqcA8QSvYF2jIQbBrYtm1HM3hrZQZrrbj00ktx8MEHY/PNN482rC3HbLa2xHOdSilhWRamTp2Khx9+eKP9jWOOOabkHdivYgZOTR5a3/eRzWa5iWwJrrcku/XWW8Wtt96KKVOm6B//+McYPHhwtDrPzExTSkWbtYdhGM2+Nj9jBqc3pfj9ZDrn5vvmWWJZFiZPnoy77rqr6q6jgw46SDc0NESdVHM/t/68Wpnry3Vd+L6P+vr6aNPTgw8+GOeddx4rmyoX30vL1E11dXXo06cPpk+frmfPns2BWqooZ599tt5ss82K6rhy1q+mfex5HlKpFG699VaepBrzgx/8AACivkKS2tJmAoxpA37++efo2LEj/vrXv+LnP/954uv/O++8U/i+r2+66Sa0b98eAJDL5YrK2YyL2LYdtb1Loa2VxVprpFIpDB48GP3799dLly7lM3Y9jWoPfd5+o9DJ/xwBmuAIDdfTsLUFbUno9S5ZCUtJABIKAkpqeFYACIVUICGUjVA5UFLAFzl4UBB17fHQq2/hZ8/8jyeGKh4DIESUCG+//ba47LLL9O9///tow/EkBAPiwQ/f9xEEAdLpNCZOnIidd95ZL168eIMP8rzzztPbb799NCgcBEGiljibgVLHcXD55Zdj3rx5bNASrr32WnHttdfisssu01OnToXruli9ejXS6TQcx4HneXBdN+okxQdrkhLoE0Jg3LhxVXl+hg4dCillUdov0zlOQv1S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i6/2MMdjUjAlJQ3JuvTt2zdK0+X7fnT8JnBTivKNp6mybRthGOI///kPFixY8I0K71//+heEEEWzn0uRwvDLgiyO42D8+PG8UWuYWR3m+z7S6TSUUhBCwPd9zJgxgwVEG+zQQw+FEAJNTU2or6+H7/tR6qkkXP9XXXUVTxIRERFVDQZAiKhq/Pa3vxVvvPEGwjBEEARrzeLO5/MACoOEpZphXO4OLIAoKPTee+/hvPPO44VCZWfuTbNvhBEPRpTy/jArx8ygZ9L3EJoyZUpUVo7jRO8lnU5H76kUTCozz/MQhiGef/75b/w7nnrqKTiOAyllSfdviqcPi69I0lrDtm2MGjWKN2qNs207ujbM9WLbNrp27Ypzzz2XsyxovfXp00dPnToVnuehrq4uqodKufpQKYV4Okqz0imfzyObzeLCCy/kZBkiIiKqGgyAEFFVufjii2FZVjSgBhQG57TWSKVS0FpHAYFa0NTUBMuyoJTCtddei7fffpsdWqIKt91225X9GMIwhOu68H0fruvCdV3cdddd3/j3vPPOO3j77bchhIgCOElYIdSvXz9ss802HOSmNh155JHo2bMnrw9aLwceeGAUXDMr4Ew7tRRMwN91XWitEYZhFIBJpVK46aabeJKIiIioqjAAQkRV5a9//at45plnog4esCatTbzTVwsrQAAgk8kgCAK88847uOaaa3iBEFW4CRMm6O7du5f9OFrvn5HL5XDTTTd94wDr22+/HdXZpm5OQgqybbfdFrvssgsvOGpTjx49MHXqVBYErZcjjzyyzXoUKM0eVPFUsUqpaOVdPp+H53nc/JyIiIiqDgMgRFR1Zs6ciSAIEARBtOLDdDJNJy/pmxxvDGZjYs/zcM011+CNN97g6g+iCjd27NhEbCJvVtGZVSDPPvvsev8u83+DIEhMcFprjaFDh/KCozblcjlMnToVvXr14ioQ+kaOPPJI3bt376I2qW3bbQZDNiWT8socQxAEcF0X8+fPx9NPP832IhEREVUVBkCIqOpcd911YtGiRUilUmhqaoq+H891nIQBxE3NpDX48MMPce6557IzS1QFRowYkZgggdlXybZtPPLII+v9e5577jlks1kIIRKz/4oQAqNHj+YFR22yLAudO3fG8ccfz8Kgb2TatGkAChNxpJRr7YNVyjowDMNoPymz/xFXCxMREVE1YgCE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cePG6ZxzztH++++vXr16ORdccIHT1hdwBg0aVBJAXV+/gup/hlue/+L7vtq3b1/1h7NWs5tvvtlmuJp25oyfynDCCSeotrY2HpPwL0uQRsd+G3KmHFAOc+bMKXm/RUv5llt0J0Tfvn01YMCAqhp0nX766dpiiy3Wmi+Yczfi4l//+pcWLlxIQBdAbJCiBCA2TjzxxLB///6xyaD0fd8ucJht3WZx+LHHHtNTTz21UQZ1kyZN0rBhw2QOs4tOKMrJZC57nqd8Pq8OHTrowgsv1FlnncXN2or3oLn/zP3ouu5GmeSYnSDRhXpTz9lxHH3wwQdauXKl3n//fa1YsULLli3TO++8oxUrVuiTTz5RtZ/h8X307du3ZFG05XVH9TP9Z8udhMViUXV1dTrssMN02223caEq0KxZs5yHHnooPProo9XY2Kj6+nrbxslkUsVikV0gMbTzzjuHRx11VGze7YlEwi7Sep6n999/X7fddhvvVVSEhQsX2j4vblzXVSaTUW1trRKJhC644AJNnz69Kq57jx49wn79+pWMJ6PzxjjMH4vFohKJhF566SUeFACxwugcQGxceumlqqmpUVNTU8mZH+XS8iyOTCajuro6rV69WjfeeONG+zkLFixwHnjggXDMmDG2dngcFk9c11WhUJDnebbMx/HHH68JEyaE8+bNY5K+iR1xxBHKZDIlu47M5Mb82fdtX/M9CAI1NTXp008/1dtvv03bfg9Dhw4N27dvz4Vo40yZOPO8uq6rVCpln7v99tuPi1TBxo4dq/79+yuVStmAh1mEIvgRT6eccoo6dOhgz96IA5NcUywWNWXKFBoJFWPp0qX64osvtNlmm9mxafR9V27mvEjf93XEEUfo8MMPDx9//PGKH99edtllJdfc7M4uxzlGX8dxHD3//PM8KABihRE6gFg44YQTwh49ekiSampqYvXZzOKGGUy//PLL+sc//rFRR5m33367TjjhBG299daS4rMLpOVZER07dtQll1yi4cOHc9NuYhv7HkOr9WU2u3d9k0K0gQF2ZBHc7KqKLo5vs802OvLII8NHH32UG6ICPffcc8706dPD4447Tk1NTfI8z2bzf9/gNDaNkSNHqlAoxKJ9zNiqUCjYDPq7776bRkLFePHFF50333wz7N27d+x2veXzeaVSqZJgzMUXX6zHH3+8oq/5KaecEh53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AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAVLb/H533qHxj1NhyAAAAAElFTkSuQmCC"

# Paleta de cores oficial
COR_LARANJA = "#FB4710"
COR_PRETO = "#000000"
COR_BRANCO = "#FFFFFF"
COR_CINZA = "#666666"

# CSS customizado para refinar a aparência além do tema base
CSS_CUSTOMIZADO = f"""
<style>
    /* Esconde menu e footer padrão do Streamlit (mas mantém o header
       pra preservar o botão que abre/fecha a sidebar) */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}

    /* Header transparente: invisível visualmente, mas funcional
       (o botão de toggle da sidebar continua clicável) */
    header[data-testid="stHeader"] {{
        background-color: transparent;
    }}

    /* Container principal: respiro no topo */
    .main .block-container {{
        padding-top: 2rem;
        max-width: 900px;
    }}

    /* Título principal: estilo Expanzio (caixa alta, peso forte) */
    h1 {{
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        text-transform: uppercase;
    }}

    /* Botão primário: laranja Expanzio com hover suave */
    .stButton > button[kind="primary"] {{
        background-color: {COR_LARANJA};
        color: {COR_BRANCO};
        border: none;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s ease;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: #E03D0A;
        transform: translateY(-1px);
    }}

    /* Botão de download: mesmo estilo do primário */
    .stDownloadButton > button {{
        background-color: {COR_LARANJA};
        color: {COR_BRANCO};
        border: none;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .stDownloadButton > button:hover {{
        background-color: #E03D0A;
    }}

    /* Caixa de upload com borda laranja */
    [data-testid="stFileUploader"] section {{
        border: 2px dashed {COR_LARANJA};
        background-color: #0F0F0F;
    }}

    /* Barra de progresso na cor laranja */
    .stProgress > div > div > div > div {{
        background-color: {COR_LARANJA};
    }}

    /* Caption em cinza da marca */
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {COR_CINZA} !important;
    }}

    /* Sidebar: fundo um pouco mais claro pra contraste */
    [data-testid="stSidebar"] {{
        background-color: #0A0A0A;
        border-right: 1px solid #1F1F1F;
    }}

    /* Logo na sidebar centralizada */
    .logo-sidebar {{
        text-align: center;
        padding: 0.5rem 0 1.5rem 0;
    }}
    .logo-sidebar img {{
        max-width: 180px;
        width: 100%;
        height: auto;
    }}
</style>
"""

st.markdown(CSS_CUSTOMIZADO, unsafe_allow_html=True)

# Logo na barra lateral
with st.sidebar:
    st.markdown(
        f'<div class="logo-sidebar">'
        f'<img src="data:image/png;base64,{LOGO_BASE64}" alt="Expanzio+"/>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Título principal
st.title("Renomeador Inteligente de Arquivos")
st.caption("Renomeie seus arquivos automaticamente usando IA.")

# Tenta carregar a chave automaticamente (Secrets ou variável de ambiente)
api_key = carregar_api_key()
chave_embutida = bool(api_key)

# ---- Barra lateral ------------------------------------------------------
with st.sidebar:
    st.markdown("### Status")

    if chave_embutida:
        st.success("✅ Chave da API configurada")
    else:
        # Fallback: se não houver chave configurada, pede ao usuário
        st.warning("⚠️ Chave da API não configurada")
        api_key = st.text_input(
            "Chave da API Gemini",
            type="password",
            help="Configure GEMINI_API_KEY nos Secrets do Streamlit.",
        )

    st.markdown("---")
    st.markdown("**📐 Padrão Expanzio:**")
    st.code("CLIENTE-000-NOME ABREVIADO-REVISÃO", language=None)
    st.caption("Ex: DIA-003-NOT. EXIG. PUBLI. 29-23-R00")

    st.markdown("---")
    st.markdown(
        "**Formatos suportados:**\n"
        "- PDF\n- TXT\n- PNG\n- JPG / JPEG"
    )
    st.markdown("---")
    st.caption(f"Modelo: `{MODEL_NAME}`")

# ---- Área principal: upload --------------------------------------------
arquivos_enviados = st.file_uploader(
    "Selecione os arquivos para renomear",
    type=TIPOS_ACEITOS,
    accept_multiple_files=True,
)

# =========================================================================
# CAMPOS DO PADRÃO EXPANZIO
# =========================================================================
# O usuário define CLIENTE, número inicial e revisão.
# A IA gera apenas a parte central (NOME ABREVIADO) baseada no conteúdo.

st.markdown("### Padrão de nomenclatura")
st.caption("Formato final: `CLIENTE-000-NOME ABREVIADO-REVISÃO.ext`")

# Cliente — obrigatório
cliente_raw = st.text_input(
    "Cliente",
    placeholder="Ex: DIA, BK, SMART, GUSTAVO NINOMIA, TOLLSTADIUS",
    help=(
        "Abreviação do cliente, conforme padrão Expanzio. "
        "Para corporativos: sigla (DIA, BK, SMART). "
        "Para residenciais: nome + sobrenome ou só sobrenome marcante."
    ),
)

# Higieniza o cliente: mantém letras/números/espaços, vira maiúsculas, sem acentos
def _formatar_cliente(texto: str) -> str:
    import unicodedata
    if not texto:
        return ""
    # Remove acentos
    nfkd = unicodedata.normalize("NFKD", texto)
    sem_acento = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Mantém letras, números e espaços; vira MAIÚSCULAS
    limpo = re.sub(r"[^a-zA-Z0-9 ]", "", sem_acento).upper()
    # Colapsa espaços múltiplos
    limpo = re.sub(r" +", " ", limpo).strip()
    return limpo

cliente = _formatar_cliente(cliente_raw)

if cliente_raw and cliente != cliente_raw.upper():
    st.caption(f"✏️ Cliente formatado: `{cliente}`")

# Linha com dois campos: número inicial e revisão
col1, col2 = st.columns(2)

with col1:
    numero_inicial = st.number_input(
        "Número inicial",
        min_value=1,
        max_value=999,
        value=1,
        step=1,
        help=(
            "Numeração sequencial. O primeiro arquivo recebe este número, "
            "o segundo recebe +1, e assim por diante (001, 002, 003...). "
            "Use um número maior se a pasta já tem arquivos."
        ),
    )

with col2:
    revisao = st.text_input(
        "Revisão",
        value="R00",
        max_chars=5,
        help=(
            "R00 = primeira versão. Apenas incremente (R01, R02...) "
            "se os arquivos forem revisões já solicitadas."
        ),
    )

if arquivos_enviados:
    st.info(f"📎 {len(arquivos_enviados)} arquivo(s) selecionado(s).")

# Avisos para o usuário antes de processar
botao_desabilitado = not (arquivos_enviados and api_key and cliente)

if arquivos_enviados and not api_key:
    st.error(
        "⚠️ A chave da API não foi configurada pelo administrador. "
        "Contate o responsável pela aplicação."
    )

if arquivos_enviados and not cliente:
    st.warning("⚠️ Informe o nome do cliente para continuar.")

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

            # Callback pra mostrar mensagens durante esperas de rate limit
            placeholder_status = log.empty()

            def avisar(msg):
                placeholder_status.info(msg)

            # 1) Chama a IA pra gerar APENAS o NOME ABREVIADO
            nome_abreviado = gerar_nome_com_retry(
                conteudo_bytes=conteudo,
                nome_original=arquivo.name,
                api_key=api_key,
                callback_status=avisar,
            )

            # 2) Monta o nome final no padrão Expanzio:
            #    CLIENTE-000-NOME ABREVIADO-REVISÃO.ext
            extensao = os.path.splitext(arquivo.name)[1].lower()
            numero_atual = int(numero_inicial) + (indice - 1)
            novo_nome = montar_nome_padrao_expanzio(
                cliente=cliente,
                numero=numero_atual,
                nome_abreviado=nome_abreviado,
                revisao=revisao,
                extensao=extensao,
            )

            # 3) Garante que não há nomes duplicados (raro, mas pode acontecer)
            novo_nome = evitar_nome_duplicado(novo_nome, nomes_ja_usados)

            # 4) Salva o arquivo com o novo nome na pasta temporária
            caminho_destino = os.path.join(pasta_temp, novo_nome)
            with open(caminho_destino, "wb") as f:
                f.write(conteudo)

            resumo["sucesso"] += 1
            placeholder_status.success(f"✅ `{arquivo.name}` → `{novo_nome}`")

        except Exception as erro:
            # Não interrompe o processamento — apenas avisa e segue
            resumo["erro"] += 1
            mensagem = str(erro)

            # Mensagem mais amigável para erros conhecidos
            if "429" in mensagem or "quota" in mensagem.lower() or "rate" in mensagem.lower():
                texto_amigavel = (
                    "Limite de requisições da API atingido mesmo após várias tentativas. "
                    "Espere 1 minuto e tente novamente."
                )
            elif "401" in mensagem or "API key" in mensagem:
                texto_amigavel = "Chave da API inválida ou sem permissão."
            else:
                texto_amigavel = mensagem

            with log:
                st.error(f"❌ `{arquivo.name}` — {texto_amigavel}")
                # Mostra a mensagem técnica completa pra ajudar no diagnóstico
                with st.expander("🔍 Ver detalhes técnicos do erro"):
                    st.code(mensagem, language="text")

        # Pausa entre arquivos pra respeitar o limite de requisições do Gemini
        # (só pausa se ainda tem mais arquivos pra processar)
        if indice < total:
            time.sleep(PAUSA_ENTRE_ARQUIVOS)

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
