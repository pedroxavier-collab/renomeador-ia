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
LOGO_BASE64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCAHqBkADASIAAhEBAxEB/8QAHQABAAICAwEBAAAAAAAAAAAAAAgJBgcCBAUBA//EAFoQAQABAwICAQsMDwUHAwQDAQABAgMEBQYHERIICRMYITE3UWF1syJBVnFygZGUlbLR0xQVFhcjMjM1NlJVV3OCsThCk6GiJDRDYnSS0lN2gyVEo8FFY8LD/8QAGwEBAAMBAQEBAAAAAAAAAAAAAAQFBgcDCAL/xAA4EQEAAQIDAgkMAgIDAAAAAAAAAQIDBAURU3IGEhYxNJGx0eEHExUXNVFhcZKhssFBgRSiIVTi/9oADAMBAAIRAxEAPwCGQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAyPhptW/vffekbUxsu1h3tTyIsUXrlM1U0TMTPOYjuz3kkO0i3P7OdH+KXfpBEwSz7SLc/s50f4pd+k7SLc/s50f4pd+kETBLPtItz+znR/il36TtItz+znR/il36QRMEs+0i3P7OdH+KXfpO0i3P7OdH+KXfpBEwSz7SLc/s50f4pd+k7SLc/s50f4pd+kETBLPtItz+znR/il36TtItz+znR/il36QRMEs+0i3P7OdH+KXfpO0i3P7OdH+KXfpBEwSz7SLc/s50f4pd+lrPqgeAGrcINC03VdR3Dg6nRn5VWPTRYsV0TTMUTVznpe0DTAM64I8PaeJu9Kdq29w4ejZl6zXcxasm1VXTfqp7s246Peno9Kr+WQYKJZ9pFuf2c6P8Uu/SdpFuf2c6P8AFLv0giYJZ9pFuf2c6P8AFLv0naRbn9nOj/FLv0giYJZ9pFuf2caP8Uu/Sj/xm4d6vwv31k7V1i5RkV27dF6xk26Zpov2qo7ldMT3e/ExPlpkGGAAAAAADlboruV026Kaqq6p5U00xzmZ8UJS6J1Fm8M7R8PNzN1aXp+TfsUXLuLcxrlVViqqOc0TMTymY708vXBFgSz7SLc/s50f4pd+k7SLc/s50f4pd+kETBLPtItz+znR/il36TtItz+znR/il36QRMG6uPnALI4Q6Bhajqu8NNz8rOyOxY+FYx66bldMRzrr5z3Ipp9THt1Q0qAD2ti7fu7r3lo+2rGRRjXdUzbWJRerpmaaJrqimJmI7sxHMHiiWfaRbn9nOj/FLv0naRbn9nOj/FLv0giYJZ9pFuf2c6P8Uu/SdpFuf2c6P8Uu/SCJgln2kW5/Zzo/xS79J2kW5/Zzo/xS79IImCWfaRbn9nOj/FLv0naRbn9nOj/FLv0giYJXX+ok3fTy7BvPQrnc/v2L1Pd96JY/rPUc8VcOapwcnb+pRHei1mVW6p96umI/zBHEbD3dwS4rbVt13dY2Nq9Fiju1X8e1GRbiPHNVqaoiPb5NfXLdduuqi5TVTVTPKqmY5TE+WAcQAAAAAAAAAAABsHgRws1Xi1vG9t3S86xgdgxK8q9k37dVVFFNNVNMRyj15mqP828u0i3P7OdH+KXfpBEwSz7SLc/s50f4pd+k7SLc/s50f4pd+kETBLPtItz+znR/il36TtItz+znR/il36QRMGyePvCPV+EO48HSNT1DH1GjNxfsizkWKJopnlVNNVPKe7ziYj4Ya2AAAAAAAAAG6ep/6n3VuL+3tR1jTtxYOmUYOXGNVbv2K65qmaIq5xNPrd1srtItz+znR/il36QRMEs+0i3P7OdH+KXfpO0i3P7OdH+KXfpBEwSz7SLc/s50f4pd+lpHjzwm1vhHujG0bVsqznWsvGjIx8uzRNNFyOcxVTynuxNMx3fbjxg10AAAAAAO1pGFd1LVcTTrFVFN3Kv0WaJqnlEVVVRTHPyc5SOnqMOJnP8APW2fjF36sEZxJjtMOJn7a2z8Yu/VnaYcTP21tn4xd+rBGcSY7TDiZ+2ts/GLv1Z2mHEz9tbZ+MXfqwRnHpbo0fJ2/uTU9BzK7deTp2XdxbtVuZmia7dU0zMTPrc4eaADtaTh1ahqmLg0VxbqyL1FqKpjnFM1VRTz/wAwdUSz7SLc/s50f4pd+k7SLc/s50f4pd+kETBLPtItz+znR/il36TtItz+znR/il36QRMEs+0i3P7OdH+KXfpO0i3P7OdH+KXfpBEwSz7SLc/s50f4pd+k7SLc/s50f4pd+kETBLPtItz+znR/il36TtItz+znR/il36QRMEs+0i3P7OdH+KXfpO0i3P7OdH+KXfpBEwSz7SLc/s50f4pd+k7SLc/s50f4pd+kETBLKvqJNy0UzVXvvRqaYjnMziXYiI+FFjWMaxhatmYeLmUZtixfrt28mimaab1NNUxFcRPdiJiOfKfGDqAADa3U9cFtS4xZOs2NO1vE0udLos11zftVV9k7JNcRy6Pe5dCfhbd7SLc/s50f4pd+kETBLPtItz+znR/il36TtItz+znR/il36QRMEs+0i3P7OdH+KXfpYLxx6mrWuFex53Vn7m0/UbMZVvG7DZx66KudfS7vOe53OiDQwAAAAAAAA72h6Rquuaja03RtNy9Rzb08rePi2arlyqfJTTEykPw66jziDr1FrK3Rm4W2MavlM26/9oyeXuKZ6Me/UCNT7ylYbtHqPuFekUUVavVq+v34mJqnIyexW5nyUW4ieXkmZbI0fglwl0nl9h8Ptv8Ac9e9iRfn4bnSBVSLdLWwNi2rcW7Wy9uUUR3qadLsREf6XXzuGfDrOiqMrYm2Ls1RymatKs8+Xt9HmCpTlPLnyfFnWv8AU18GdYomK9mY2FXM8+yYN65YmPepq5f5NQ756ibSL1Ny/svd2XiXO/Tjanai7RM+LslHKaY/lqBCMbK4o8D+JHDqK8jXtv3bmnU//wAhhz2fH9+qO7R/NENagAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA2b1K/8AaE2X5zp+bUtLhVp1K/8AaE2X5zp+bUtLgAAAAAAAAAAAABFHrkvg/wBredbnoZSuRR65L4P9redbnoZBBV6G29Z1Db2v4GuaTkVY+dgX6Mixcpn8Wumece9448TzwFt3CXeun8QuH2k7s06qiKM6xE3bUTzmzejuXLc+5qiY9rlPrsqQL6gDidOg7yv8PtUyJjTtbq7Jg9Kr1NvLiO9/PTHL26afGnoAAAjd1enDad1cNqN36dj9k1TbvSuXIpj1VzEq5dkjy9CeVfkjpJIvzyrFnJxrmNkW6btm7RNFyiqOcVUzHKYmPFMAprGxOqK4e3uGnFbVdvRRX9r6q/snTrlUfj49czNHtzT3aJ8tLXYAAAOdm3cvXaLVq3VcuV1RTTRTHOapnvREevIN99Q/w2+7bivb1vPx+yaPtzo5d3pU86bmRz/A2/L3YmufJR5VjTWPUy8N6OGXCjTtFvW6Y1XJj7M1OuIjnN+uI50e1RHKj+WZ9ds4AABxu3KLVqq5crpoooiaqqqp5RER35lyR46ufibGzOGc7Y03I6Gs7ipqsR0KuVVrFjuXa/J0ufQj26vECIfVR8Sq+JvFbP1PGv1V6NgzOHpdPP1M2aZnnc5eOurnV7U0x6zVQAM76nzw57I8+4npaWCM76nzw57I8+4npaQWugAAAAAAAAAcoYPxD4ScPN+2LlG5tr4GTfrieWXbo7FkUz44uUcqvemZjxxLOAEFuMvUd63pFq9qvDrPr1vFp51Tp2VNNOVTH/JVHKm57XqZ8XNFjUcLM03OvYGoYt7EyrFc271m9RNFduqO/FVM92JXINUcfeBm0+K+l3LmVZo07cFu3yxdVs247JEx3qbkf8SjyT3Y9aYBV8Mo4m7D3Jw73Vkbd3NhVY+TannbuU85tZFvn3Llur+9TP8Al3p5THJi4AAAAAAAPtMTNUREc59aPGCc/W49qTh7K1/eF+1yr1LLpw8eqefPsVmOdUx5Jrr5fyylgwbgHtaNmcHts7dqtxbvY+DRXkREcvw1z8Jc/wBVUs5AAAABF/rie1Ptnwv0rdNmiJvaLn9juzy/4N+Ipn4K6bfwygOtt4wbYjefDDcW2ej0rmfgXbdnyXYjpW5/7opVKXKaqLlVFdM010zMVRMcpifXBxAAAAAAABO7rbfg23N54p9DQlUir1tvwbbm88U+hoSqAAAaW6sbhv8AfB4RZdzCx+y6zonSz8Hox6quIj8Lbj3VEdyPXqppbpJjnHIFM43J1XnDarh1xczacPH7HousTVnadNMcqaYqn8Jaj3Fcz3P1Zp8bTYAAAAPa2H+m+hecsf0tK35UDsP9N9C85Y/paVvwAAAAKluNXhg3l59zfTVsQZfxq8MG8vPub6atiAD1tnfpZpH/AF1j0lLyXrbO/SzSP+usekpBcFAQAAAAAAAAAAAAA0v1ZO/fuG4K6jTi3oo1PWZ+1uJET6qOnE9kr96jpd3xzT41Z8pC9Xfv2d1cX69v4l3padty3OJHKe5VkVcqr1XvT0aP5J8aPQAAJhdbS/Ou9/4GF8+8mshT1tL8673/AIGF8+8msAAAjz1wLwA1edsX+laQyPPXAvADV52xf6VgrtAAAAAASA6nfqZ9ycR6MfXtfrvaFtmv1VF2qj/aMun/APqpnvU/89Xc8UVM66kDqb6NYt4m/wDiBgxVp1URd0zS71P+8eK9dj9T16af73fnucombluim3RFFFMU00xyiIjlEQDFeG3DnZ3DvSKdN2pouPg08oi7e5dK/fnx13J9VV/SPWiGWAAAAAAADjct0XLdVu5TTVTVHKqmY5xMeKUduOvUq7Q3pbyNW2hRY2zr1UTV0bVHLDyKv+e3H4kz+tR7cxKRYCoff2zNybF3Hf0DdGl3tPzrXd6NfdpuU+tXRVHcqpnxx/Vj62HjJww2xxR2rc0TcGLEXaYmrDzbdMdmxLn61E+Lx0z3Jj3pis/i9w63Dwx3jf25uGxHTp9XjZNuJ7FlWpnuXKJ8XjjvxPOJBhwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANm9Sv/aE2X5zp+bUtLhVp1K/9oTZfnOn5tS0uAAAAAAAAAAAAAEUeuS+D/a3nW56GUrkUeuS+D/a3nW56GQQVAB2NOzMnT8/Hz8K9XYysa7Tes3aJ9VRXTMTTVHliYiVqfAPiDjcTOGGlbntzRTl10dgz7VP/CyaIiLkcvWiZ5VR5KoVSJE9QtxN+47iX9y2pZHQ0fcdVNjnVPqbWVH5KrydLnNE+3T4gWHgAAAjr1dnDP7sOGkbp03H6Wrbciq/V0aedV3Enu3ae53+jyiuPaq8avFcrkWrd+xcsXrdNy3cpmmuiqOcVRMcpiY8UqsuqR4eXOGnFjVNAot1U6bcq+ytNrnvVY9czNMc/HTPSony0g1uAAkF1DXDb7tOKlO4NQx+npG3OhlV9KnnTcyZn8DR70xNc+5jxtAWLVy/fos2aKrly5VFNFFMc5qmZ5REeVaX1NnDqzw04UaZoVdqmnU71P2VqdyO/VkVxE1Rz8VMcqI9z5QbKjuAAAA/DPy8bAwb+bmXqLGNj26rt25XPKmiimOc1T5IiJVWcfuIWTxN4n6pua5NdOHVX2DT7NXc7FjUc4ojl457tU+WqUuOr/4m/c/suxw/0zI6Oo67T2TN6M+qt4dM97ydOqOXtU1R66BIAADO+p88OeyPPuJ6WlgjO+p88OeyPPuJ6WkFroAAAAAAAAAAAAANdcfeFGicWNmXdI1CmjH1KxFVzTc/o86sa7y/zoq5RFVPrx3e/ESrC3ht3Vtp7mz9ua7izi6jgXps37c92ImO9MT69MxymJ9eJiVwSLPV8cKbeu7Tp4jaPj0/bPR6Io1CKae7exOf40+Obczz9zM+KAQNAAAAAAbB6nTan3Z8adsaFXa7JjV51F/Kp5c4mza/CVxPkmKeXvtfJcdbg2n9lbm3HvPItc7eFjUYGNVMdybl2elXMeWKaKY9qsE4IAAAAAAnvKtOqj2tG0OOu59Lt2+x417LnMx49bsd6OyRy8kTVMe8tLQq65HtWLep7Y3pZtxHZrVzTcmqPHTPZLf+VVz4IBDwAAAAAAAE7utt+Dbc3nin0NCVSKvW2/BtubzxT6GhKoAAAAGmuq/4b/fE4Q5sYVnp6zo/Sz9P5d+uaY/CWv5qOfKP1opVmzHKeS5dWf1YfDeOHvF3LqwcebWi61zz8DlHKmjpT+EtR7mvnyj9WqkGlwAAAe1sP9N9C85Y/paVvyoHYf6b6F5yx/S0rfgAAAAVLcavDBvLz7m+mrYgy/jV4YN5efc301bEAHrbO/SzSP8ArrHpKXkvW2d+lmkf9dY9JSC4KAgAAAAAAAAAAAYhxl3pjcP+GeubryJpmvCxpnHoqn8pfq9Tao9+uY97my9Cvri+/pu5ujcOsG/PQsR9sdRimZ7tcxNNmifajpVcv+anxAiFn5eRnZ1/NzLtV7JyLlV27cq79ddUzNUz5ZmZl+AAAAmF1tL8673/AIGF8+8mshT1tL8673/gYXz7yawAACPPXAvADV52xf6VpDI89cC8ANXnbF/pWCu0AAABILqMuDFPEbd1W4twYvT2xo9ymblFX4uZkd+m15aY7lVXk5R/e7mjdt6Pn7g1/A0PS7FV/Oz8ijHx7cf3q65iI97urX+FGy9N4fbA0naemUU9jwrEU3bkRym9dnu3Lk+WqrnPtco9YGUW6KbdFNFFMU00xyiIjlEQ+gAAAAAOM10x36oj255PsVRPenmD6AAAA1p1RXCnTOK+wr2kXoos6tjRVe0vLmO7ZvcvxZ/5KuURVHtT34hssBTnrml5+iaxmaRqmLcxc7DvVWMizcjlVbrpnlMT78OmmD1wvhlTi5uDxN0rG5UZVVOHq3Qp71yI/BXZ9uImiZ8lPjQ+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABs3qV/7Qmy/OdPzalpcKtOpX/tCbL850/NqWlwAAA/PJu9hx7l6Y6XQomrl4+Uc36OvqX5uyf4VfzZBEyvq4NGprmn732f3JmPzlR9W+dvDo37vc/wCUqPq0Jr/5av3U/wBXAE3O3h0b93uf8pUfVnbw6N+73P8AlKj6tCMBNzt4dG/d7n/KVH1Z28Ojfu9z/lKj6tCMBNzt4dG/d7n/AClR9WdvDo37vc/5So+rQjATc7eHRv3e5/ylR9W1D1T3VA4HGHbmk6VibZydIqwMurIm5dyqbsVxNE08uUUxy77QIAAA5Wrldq5Tct11UV0zFVNVM8piY70xPjcQFovUvcSqOJvCnA1XIuxOsYf+x6pRz7vZqYjlX7VdPKr25mPWbTVq9RvxOnh5xWsY2fkRb0PXJpws3pT6m3VM/grvk6NU8pn9WqpZVAAACPfVzcNPuz4XVbj07Hm5rO3YqyKYpj1V3Gn8tR5eURFce5mPXSEcb1u3etV2rtum5brpmmqiqOcVRPcmJj14BTSNmdUvw6ucM+LGp6HatV06XkT9l6ZVMdyceuZ5U8/XmmYqo/l5+u1vjWL2TkW8fHtV3b12uKLdFEc6qqpnlERHrzMgkH1CvDad4cUI3PqGNNekbcmnI51U+puZU/kqfL0eU1z7mnxrEWuup04e2uGnCrStvVUUfbCqj7J1KumPx8muImuOfrxT3KI8lLYoAADz9x6xp239Bz9c1bIpxsDBx68jIu1f3aKY5z7/AJPXl6CIXXCuJs4el4fDLSsjley4pzNVmie7FqJ/BWp91MdKY8VNPjBFDi5vbP4h8QtW3ZqHOirNvTNm1z5xZsx3LduPapiPf5yxMAAAGd9T54c9kefcT0tLBGd9T54c9kefcT0tILXQAAAEMLnVw36LlVP3ubc8pmPzvP1SZ895TXkfl7nup/qCYnbx3/3cW/lefqjt47/7uLfyvP1SG4Ccei9W9tu9dpp1jYurYdEz3asXMt5Ex71UUNxcOeqB4Wb6yLWHpe5LWJn3ZiKMPUafse7VM+tT0vU1T5KZlV0AuYFf3UvdUvq+z9Rxdr75zr+o7ZuzFu3lXpmu9p8z3p6XfqtR69M85iO7He5TP3GvWcnHt5GPdou2btMV27lFUVU1UzHOJiY78THrg/QAB+GoYmNn4GRg5lmi/jZFqq1etVxzproqiYqpnyTEzD9wFTHGnZd7h/xO1zal3pzawsmfsaurv12KvVW6vfpmPf5sOTA65DtOizq+2t649qmn7JtV6dlVR3OdVHq7fPxzyquR/LCH4AAAACy/qLdq/cxwC0S5ctdDJ1fpane9TymYufk//wAcUT76unZGhZG594aRt3Fiezalm2sWnl63Trinn70TMrdtKwrGm6Xi6di0RRj4tmizapjvU0U0xTEfBEA7IAAAAADUHVibW+6rgBuG1bomvK023TqVjl47M86/htzXHvw2+/HPxbObhX8PJoiuzft1WrlM/wB6mqOUx8EyCm0e9xC2/f2rvnW9t5FMxc03OvY3d9eKa5imfamOU++8EAAAAAAE7utt+Dbc3nin0NCVSKvW2/BtubzxT6GhKoAAAABp7queG0cRuEmbaw8eLutaTFWdp0xHqqqqY9Xaj3dPOOX60UtwkgpnnvjdvVlcN54f8Xcq/hWOx6NrnSzsLl+LRVM/hbX8tU84jxVUtJAAA9rYf6b6F5yx/S0rflQOw/030Lzlj+lpW/AAAAAqW41eGDeXn3N9NWxBl/Grwwby8+5vpq2IAO5ouZGn6xh51VE3Ix8i3dmmJ5TV0aoq5c/edMBNvt4dG/d7n/KVH/g+9vDo37vc/wCUqPq0IwE3O3h0b93uf8pUfVnbw6N+73P+UqPq0IwE3O3h0b93uf8AKVH1Z28Ojfu9z/lKj6tCMBNzt4dG/d7n/KVH1buaH1Z2HrWs4Wkabw31G/mZt+jHsW41Kj1VddUU0x+T8coLpM9b92D90HEzK3hm2ZnB2/a52JmO5VlXImKf+2np1eSZpBP6nnNMdKIirl3Yiecc30AAAdPXNTw9G0bM1bULsWcPDsV379ye9TRRTNVU/BCpfibuvN3xv7Wt15/OLupZVV6KOfPsdHeoo9qmmKY95N/q/wDfs7e4X2No4V/oZ24bvRu9Ge7Ti25ia/8Auq6FPljpK/AAAAATC62l+dd7/wADC+feTWQp62l+dd7/AMDC+feTWAAAR564F4AavO2L/StIZHnrgXgBq87Yv9KwV2gAAR3wSg63nsiNa4jajvLKsxVjaDjxRjzVH/3F6JiJj2qIr9rpQnw0T1C+26NB4A6dmTbim/rORez7k+vMTV2Oj/TRHwt7AAAA091WfFSvhdwyuZGm3Kade1WqcTTefd7HPLnXe5ev0ImOX/NVSDxuqI6pXbfDK9d0HSLNGvbmpjlXj03OVnEn1uzVR3el6/Qju+Oae4hnvzqgOLG8btz7O3dm4OLX3sTTapxbUR4vUcqqo91MtZZWReysm7k5N25evXq5uXLlyqaqq6pnnMzM9+Znu835A717V9UvXJuXtSzLlc9+qq/XMz7/ADehoW893aDkU5Gi7n1nTrtPeqxs25b/AKS8EBJrhL1X+9dByrGJvi1RuXTOcU136aKbWXbp8cTHKmvl4qoiZ/WTe4f7y27vvbOPuHbGo287Bv8Ac509yq3XHforp79NUevE/wBO6qHba6l/i1ncLOIOPfvZFydvahcpsatj8+dPQmeUXoj9ajnz8sc49cFnw4WLtu/ZovWblNy3XTFVFdM84qiY5xMT4nMAAGN8UNp4m+Nga1tXM6MW9RxK7VNdUc+x3OXOiv8AlqimfeVJ6nh5Onajk6fmWptZONers3qJ79NdMzTVHvTErkJ7sKzurQ21Ttvqgtdi1R0LGpxb1G1EU8o/C0+r/wBdNYNMAAAAAAAAOUUVzHOKKpj2nFJXhL4O9I/h1+kqVuaZh/gWoucXjazpz6d6+4P5J6YxFVnj8XSNddNf5iPfHvRt7Hc/9Or4Dsdz/wBOr4EvO745O745UXKqdl/t4Nd6u4/7H+n/AKRD7Hc/9Or4HyaK4jnNFUR5YS97vjlg/G/wfZX8ez8574bhJN+9Tb83prMRz+CJj+AsYTDXL/n9eLEzpxdNdI3keAGoc+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAbN6lf+0JsvznT82paXCrTqV/7Qmy/OdPzalpcAAAOvqX5uyf4VfzZdh19S/N2T/Cr+bIKcr/5av3U/1cHO/wDlq/dT/VwAAAAAAAAAAAAAWWdRxxNjiHwox8fPyZua5ocU4Wd06uddymI/BXZ91THKZ/WpqVptqdS7xKr4ZcV8DVMm7NOjZv8AseqU8+52Gqe5X7dFXKr2omPXBaIONq5Rdt03LddNdFURNNVM84mJ70xPicgAAaB6uHht923Ci5rmBjdk1jbvSy7XRj1VzH5fhqPL3IiuPceVHPqEOGn3W8Sat26lZivSduTTdoiqO5dyp59jp/l7tc+WKfGsIuUU3LdVuumKqKo5VUzHOJj14Yzwx2Jt/h3turQNuY82cSvKvZVXS5c5ruVc+7y9aI6NMeSmAZRHcAAAB4m/Nzabs3Z2qbo1e50MLTceq/c7vKauX4tEf81U8qY8swqe37ufUt57y1TdGr3JrzdRyKr1zu84pie5TRHkppiKY8kQlH1wridGXqOHwy0q/E2cWaczVppnv3Zj8Fan3MT05jx1U+JEAAAAABnfU+eHPZHn3E9LSwRnfU+eHPZHn3E9LSC10AAACe8pryPy9z3U/wBVyk95TXkfl7nup/qD8wAAAE/Ot/8AEW9uPYGXsrU703M3b80zi1VTzmrEr59Gn+SqJj2ppj1kA28Oof3DXoXVC6NjzXNNjVrV7Aux4+lRNdH+uin4QWUBHdjmAAA0b1cmhxrHU86xkRTE3dLv2M2jud3uXIoq/wBNyfgVsrZ+OGnRq3BzeOnzT0pvaJlxT7qLVU0z8MQqYkAAAAEh+oE2pGu8bqdbvW4qx9Bw7mTznvdlrjsduP8AVXP8sLEUZut5bU+1PCbP3Nft8r+uZ9XY55d+xZ9RT/rm5/kkyAAA0brvEucXqwtD2JGRVGJd0G7Zu0T+L9k3J7NT7/QtREe6nxt5T3u6q63dxGvX+qZyuIuNd50Y+4KcixMT3KrFq5FNMe1Nunu+3ILRR+WJkWsrFtZOPci5Zu0RXbrjvVUzHOJ+CX6gAAr064FtaNF40W9es24ps67g0XqpjvTet/g6/wDKKJ99HJYH1wva3244QYe4rNHO/oWfTVVPLnys3o6Ff+qLc+8r8AAAAAABO7rbfg23N54p9DQlUir1tvwbbm88U+hoSqAAAAAABqDqtuGv3yOEmbj4Vnp61pfPO07lHdqqpj1dv+ennHtxSrHqiaappmJiY78SuXVsdWfw3jYPFvIzcHH7Fo2vdLOxOjTypouTP4a3HtVTziPWiuAaOAB7Ww/030Lzlj+lpW/Kgdh/pvoXnLH9LSt+AAAABUtxq8MG8vPub6atiDL+NXhg3l59zfTVsQAAAAAAAAB9iJmeURzlaJ1LGwo4fcGdG0y/Zi3qeZR9nahzju9muxE9Gfc09Gn3p8aCvUlbBnf3GnScTJsTc0vTavtjn84maZt25iaaJ91X0afamfEs9gAAAmeUcxqnqrN+/e/4L6xqOPei1qWdT9r8DlPKqLt2Jiao8tNEV1e3EAgr1V+/J3/xo1fULF7smm4FX2vwOU86exWpmJqj3VfTq9qYanAAAAAEwutpfnXe/wDAwvn3k1kKetpfnXe/8DC+feTWAAAR564F4AavO2L/AErSGR564F4AavO2L/SsFdoAAO3o1inK1fDxq/xbt+3RPc59yaoj/wDYLauF2kU6Dw221o1NEUfYWlY1mqIjl6qm1TFU+/POWRuFimKLNFEd6mmI/wAnMAABXh1f257ms8b50Om5M4+hYVqxFPrRcuR2Wufgqoj3lh8qq+qXy6s3j9ve9VVNU06zftc+X/p1dD//ACDXYAAABAAs36jjdVzdfADQL2Tdm5l6dFem36pnnP4GeVHOfH2ObbcKK3W3825c4bbjwKq5mmxq9Nymnu9zp2af/D/JKkAABCLrlOk029zbQ12mj1WRh38Surl/6ddNdMf/AJKk3UTeuS2Iq2RtTJmnu29SvW4q8XStc+X+kEGQAAAAAAAEleEvg70j+HX6SpGpJXhL4O9I/h1+kqZvhP0Wne/Ut35P+n3NyfypZUAwzrYwfjf4Psr+PZ+czhg/G/wfZX8ez85Oy3pdrejtVOfezL+5V2I8AOnvn8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABs3qV/7Qmy/OdPzalpcKtOpX/tCbL850/NqWlwAAA6+pfm7J/hV/Nl2HX1L83ZP8Kv5sgpyv/lq/dT/AFcHO/8Alq/dT/VwAAAAAAAAAAAAAABYh1C/E2d58NPuY1LI7JrG3KabHOqfVXcWfyVXl6PKaJ9qnn30iFUvAHiDk8M+KGl7mtzXVh019g1C1T/xcauYiuOXrzHcqjy0wtT0/LxtQwMfOw71F/GyLVN2zdonnTXRVETTVHkmJiQfuAAAAAAxXi1vXA4e8PtW3ZqPKqjCsTNq1M8pvXp7lu3HuqpiPa5yypArq/8Aib9v952OH2mZHS0/Q6uyZs0z3LmXVH4vl6FM8vbqqj1gRs3JrGobh1/P1zVsirJz8/IryMi7V/erqnnPtR3e5HrQ88AAAAAGd9T54c9kefcT0tLBGd9T54c9kefcT0tILXQAAAJ7ymvI/L3PdT/VcpPeU15H5e57qf6g/MAAABm/APKqwuNezMmiOc0a3i9zny587kR/+2EMv4K+F7aHnrE9LSC2mO8PlPe999AAB5m7LdN7a+q2a+fRrwr1NXLxTbqU91d/3oXD7l/R7Uf+ku/MlTxV3/egHwAB+mNauX8i3Ys0TXduVRRRTEd2apnlEfC/NtfqS9qxu3j1tvEuW+njYV/7YZHiimz6uOft1xRHvgsc4W7ao2dw60DbFFMUzp2BasXOU84m5FPOuffqmqWSkd4AABgnVBbnjZ/BndGvRci3ds4FduxMzy/C3Pwdvl5elXCqOZ9Vz5p29ca3R9hcP9C2pZuxTc1TOnJvUxPdm1Zp7kT5Jrrpn26UEQWkdSrueN18BdrahXem5k2MT7CyOc85iuzM2+75Zimmr34bQRF627uWb+3dz7Su3JmcTJt59imf1blPQr5e/RT8KXQAAMc4n7btbw4e6/tm7TTP2xwLtiiZ71Nc0z0KveqimfeVHZNm9j5NzHyLdVu9armi5RVHKaaonlMT5ea5Se8q+6rba33J8fNyYdu12PGzL8ahj8omImi9HTnl7Vc1x7wNUAAAAAAnd1tvwbbm88U+hoSqRV6234NtzeeKfQ0JVAAAAAAANR9Vjw4++Pwjz8TEsRc1jTOedp3KPVVXKInpW493Tzjl4+i24SCmiYmJmJjlL43t1avDX7g+LN7UtPx+x6Lr/SzMboxypt3ef4a3HtVT0ojxVx4miQe1sP8ATfQvOWP6Wlb8qB2H+m+hecsf0tK34AAAAFS3Grwwby8+5vpq2IMv41eGDeXn3N9NWxAAAAAAAAGT8K9o5e++IWi7UwulFeoZVNu5XTHPsdrv3K/5aIqn3gTg6gPYUbb4V3d15lmKdQ3Hd7JRMxHOnFtzNNuP5p6dXliafEki6ukYGJpWlYmmYFmLOJiWKLFi3HeooopimmPeiIdoAABX71wHfv3Q8T8faGHf6WDt210bsUz3Ksq5EVV+30aehT5J6XlTf4m7rw9kbB1rdedNPYdNxa70UzP5SvvUUe3VVNNPvql9d1PM1rWs3V9RvVX8zNyK8jIuVd+u5XVNVU/DMg6QAAAAAJhdbS/Ou9/4GF8+8mshT1tL8673/gYXz7yawAACPPXAvADV52xf6VpDI89cC8ANXnbF/pWCu0AB6O2f0i03/q7Xz4ec9HbP6Rab/wBXa+fALhqPxY9p9fKPxY9p9AAAlVB1QPh031/7hzvT1rX5VQdUD4dN9f8AuHO9PWDBgAAAAATj62z+h+7fOFj0VSWqJXW2f0P3b5wseiqS1AAARV65H4NdteeZ9BWlUir1yPwa7a88z6CsEEQAAAAAAAEleEvg70j+HX6SpGpJXhL4O9I/h1+kqZvhP0Wne/Ut35P+n3NyfypZUAwzrYwfjf4Psr+PZ+czhg/G/wAH2V/Hs/OTst6Xa3o7VTn3sy/uVdiPADp75/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAbN6lf+0JsvznT82paXCrTqV/7Qmy/OdPzalpcAAAOGRbi9j3LMzMRXTNPOPW5xycwETK+oi2zVXNU751fuzz/3S39L52kO2fZzq/xO19KWgCJfaQ7Z9nOr/E7X0naQ7Z9nOr/E7X0paAIl9pDtn2c6v8TtfSdpDtn2c6v8TtfSloAiX2kO2fZzrHxS19KFu6dOo0fc2q6Tbu1XaMLNvY1NdUcpqiiuaYmY8vJcNKofiT4RNy+dsr01QMfAAAAAAAAT16gDibGv7Mv8P9UyInUdDp7JhdKr1VzEqnvfyVTy9qqnxIFMr4Sb21Dh5xC0ndmnTVNeFeib1qKuUX7M9y5bn26Zn3+U+sC20dDbmsafuDQcHXNKyKcnBzrFGRj3af71FUc4n2+73nfAAAABgXH3iFi8M+GOqbmuTbqzKaOw6fZq/wCLk19yiOXiju1T5KZVWajmZWoZ+Rn5t+u/lZN2q7eu1zzqrrqnnVVPlmZlIDq6OJv3ZcS/uX03J6ejbdqqseoq503sqe5dr7nf6PLoR7VXjR3AAAAAAAZ31Pnhz2R59xPS0sEZ31Pnhz2R59xPS0gtdAAAAnvKa8j8vc91P9Vyk95TXkfl7nup/qD8wAAAGacCrFzJ4zbNsWoia6tbxOX+LSwtubqLdv16/wBURt6ehNVnTey6hen9WLdE9Gf8SqiPfBZlHeCO5HIAAB5256op25qVVUxFMYl6Zme9EdCpTzV3/ehbfxcz6dL4V7s1GqeUY2jZd3n7VmqVSEg+AAJn9bb2r0be6N637c86pt6Zi1THrflLvt/8L4JQwhaN1KW1PuQ4D7ZwLlrseVlY32fkxPf7Jf8AV8p8sUzTT/KDaQAAOvqWXZwNPyM7Jriixj2qrtyqZ5RTTTEzMz70Arv6vTdH2+475Ol2bvTxtDw7WFTET3OyTHZLk+3zrimfco/vZ3vrl/c28NY3DkzVN3Us27lVdLvx065qiPeiYh4wN59Q5uX7nuqA0vHruTTj6zYu6dc8XOqIro/10Ux78rJY7sc1PW1dXv7f3LpmuYvPs+n5drKtxE8uc0VxVy/y5Le9Hz8fVNJxNTxK4rxsuxRftVR/eorpiqJ+CYB2gAENeuSbWmqna+9LNr8XsmmZNcU+3ctc5/xfhTKaq6rHan3X8BtyYNq1NzLxLEahi8o5zFdmenMR5Zpiun+YFXg+z33wAAAAE7utt+Dbc3nin0NCVSKvW2/BtubzxT6GhKoAAB0svVcDE1XB0zIyKLeVnRcnGoq/4nY4iqqI8sRPP2onxO6jV1dO5tR2ZZ4fbo0mqKczTdcrv0RPeriLXqqJ8lVMzTPkkElR42yNyabu/aWmbl0e52TB1HGpv2pnvxEx3aZ8sTzifLEvZAABqjqquHFPEnhHqGn41iK9YwOedpsxHqpu0RPO3/PTzp9voz6yr+umaappqiYmJ5TEx3YXLq3+rY4aTsTitd1bAx4t6LuGa8zH6Mcqbd7n+Gt+T1UxVHkrjxA1DsP9N9C85Y/paVvyoHYf6b6F5yx/S0rfgAAAAVLcavDBvLz7m+mrYgy/jV4YN5efc301bEAAAAAAAEyet0bBiq9rPEXOsc+hE6dp01R688qr1ce90KeflqhD/TMLJ1LUsbT8K1Veycq7TZs26Y5zXXVMRTEe3Mwtk4Q7PxdhcN9E2nixT/sGLTTeriPyl6fVXK/frmqQZWAAD8M/Kx8HBv5uXeos4+Pbqu3blU8ooopjnVVM+KIiZBEPri+/ew4Gj8OsK/6vImNR1Cmmf7lMzTZon26ulVy/5afIhUzDjNvPI4gcTNb3XfmroZuTP2PRV/w7FPqbdPvUxHv82HgAAAAAAmF1tL8673/gYXz7yayFPW0vzrvf+BhfPvJrAAAI89cC8ANXnbF/pWkMjz1wLwA1edsX+lYK7QAHo7Z/SLTf+rtfPh5z0ds/pFpv/V2vnwC4aj8WPafXyj8WPafQAAJVQdUD4dN9f+4c709a1+VUHVA+HTfX/uHO9PWDBgAAAAATj62z+h+7fOFj0VSWqJXW2f0P3b5wseiqS1AAARV65H4NdteeZ9BWlUir1yPwa7a88z6CsEEQAHubF0S1uLdGJpF6/XYovxXM3KKYmY6NFVXen2nhsy4L+EjTPaveirRsbXVbw1yumdJiJn7J+VWaL2Os27ka01VUxMfCZjVnP3l9N/beX/g0/SfeX039t5f+DT9LagwHprHbT7R3OzclMo2Mdc97Vf3l9N/beX/g0/SfeX039t5f+DT9LagemsdtPtHcclMo2Mdc97Vf3l9N/beX/g0/S2DtfSLeg6Di6TavV3qMemaYrqjlM86pnve+9MR8TmGJxNMU3atY/pNwOS4HAVzcw1vizMac8839yAIS0Hjbx0G1uTQrmlXsivHouV0VzXRTEzHRnn3peyP3buVW64rpnSYed+zRft1Wrka0zGkx8Gq/vL6b+28v/Bp+k+8vpv7by/8ABp+ltQWXprHbT7R3KLkplGxjrnvar+8vpv7by/8ABp+k+8vpv7by/wDBp+ltQPTWO2n2juOSmUbGOue9qr7y+m/tvL/wafpaq3ZpdGi7jztLt3artGNdm3FdUcpq8qVMozcT/wBP9Z/6mf6QvcgzDE4m9VTdq1iI+HvhkOGWS4HAYW3XhrfFmatOeebSffLGgGrc5AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAbN6lf+0JsvznT82paXCrTqV/7Qmy/OdPzalpcAAAAAAAAAAASqH4k+ETcvnbK9NUt4lUPxJ8Im5fO2V6aoGPgAAAAAAAEDdnUc8M54h8WMa/n2Jr0PROjm53OnnTcqifwVqfdVRzmP1aagTU6kvaut7Q4F6Fpev5F6vLu015cY9z/AO0ou1dOm1HtRPOY9aaphtgjuQAAAOtqmPdy9NysWzl3cO7es126Mi3y6dqqaZiK459znEzzj2nZAVCcQduattLeur7c1yJ+2GBlV2r1Uzz7JPPnFcT68VRMVRPil4SbXXCuGP2VpuHxN0rHmbuJFOHq0UU9+1M8rV2faqnoTPiqp8SEoAAAAAADO+p88OeyPPuJ6WlgjO+p88OeyPPuJ6WkFroAAAE95TXkfl7nup/quUnvKa8j8vc91P8AUH5gAAAJ3db14eX9G2hqG/dSsTbv61MWMGKo5T9jUTPOv2q6/wDKiJ9eGjupg6njWeJOp4+vbhx8jTtoWqorqu1RNFzP5T+Ttc/7s+vX3o7sRznvWKafh4un4NjAwce3j4uPbptWbVunlTbopjlFMR60REA/cAAAGnerM1unROp33NPS6N3OotYNvyzcuUxVH/ZFaslNnrkW6qbOjbZ2ZZuz2TIvXNRyKInvUUR2O3zjyzVX/wBqEwAAMp4SbZuby4mbe2xRTNUahn2rVzuc+VvnzuT7UURVPvLbLFq3Zs0WbVMUW6KYpppjvREdyI+BAjrd21ftpxR1PdN610rOi4M0W6pp59G9fmaY7vj6EXPhT6AAAah6sLc87X4Abju2rk0ZGo2qdNs8p7szeno1f6Om28hl1yXc/OvauzbNXei7qWRH/wCO1/8A9f8AIENAAI76zbqNty/dL1P2367lya7+m016de51c5ibVXKj/RNCslMvrbO5uU7r2fdrnlPYtSx6efc/9O7PovgBM0AB+eTZtZGPcsX6IrtXKZorpnvVUzHKY+B+gCozintq7s/iNuDbF2Jj7XZ92xRMxy6VuKpmir36Zpn32NJL9cL2t9qOLmFuS1aimxrmDTNdUU8ud6z6ir3+jNtGgAAAAE7utt+Dbc3nin0NCVSKvW2/BtubzxT6GhKoAABE7rk/6DbU853fRJYondcn/QbannO76IHidbv4kdzUeGWp5Ef3s/SelP8AjWo/yriPdymYp/2TuLUdpbt0vcuk3Jt5um5NGRann3Jmme7TPkmOcTHimVsmwdzadvLZmk7o0mvpYepY1N+3HPu0TP41E+WmqJpnyxIPcAAat6qLhzTxK4SalpWPairVsOPs3TJ5d3s1ET6j+enpU+3MT6zaRIKg9j0V299aJbuU1UV06njxVTVHKYnstPclb4r96qHht9w/VHaPrOn4/Y9G3FqVnMsdGnlTbv8AZqOzW49b8aYriPFXy9ZYEAAAACpbjV4YN5efc301bEGX8avDBvLz7m+mrYgAAAAAACRPUGbC+6ji590mXY6en7btxk85j1M5NfOLMe3HKqv26IWHtOdR7sL7hOCml0ZNjsWp6vH2yzeccqom5Edjon3NEU9zxzLcYAACPnV279+5Pg/VoOJe6Go7juTiUxE+qpx6eU3qvfjo0fzpBqz+rJ37G+eNWo0Yl7smmaL/APTcSYnnTVNEz2SuPbr6Xd9eIgGlwAAAAAAATC62l+dd7/wML595NZCnraX513v/AAML595NYAABHnrgXgBq87Yv9K0hkeeuBeAGrzti/wBKwV2gAPR2z+kWm/8AV2vnw856O2f0i03/AKu18+AXDUfix7T6+Ufix7T6AABKqDqgfDpvr/3DnenrWvyqg6oHw6b6/wDcOd6esGDAAAAAAnH1tn9D92+cLHoqktUSuts/ofu3zhY9FUlqAAAir1yPwa7a88z6CtKpFXrkfg12155n0FYIIgAMy4L+EjTPaveirYazLgv4SNM9q96KtDzHol3dnsWmSe0sPv0/lCRoDlz6CAAAAAAAAAAAAfJRm4n/AKf6z/1M/wBISZlGbif+n+s/9TP9Iabgv0iv5fuGD8oHQre9+pY0A27koAAAAAAAAAAAAAAAAAAAAAAAAAAAAADZvUr/ANoTZfnOn5tS0uFWnUr/ANoTZfnOn5tS0uAAACZiImZmIiO/Mjr6n+bsn+FX82QeX92O0fZTofyha/8AI+7HaPsp0P5Qtf8AkqHvzPZq+9+NPreVw5z5PgBb392O0fZTofyha/8AI+7HaPsp0P5Qtf8AkqE5z5PgOc+T4AW9/djtH2U6H8oWv/I+7HaPsp0P5Qtf+SoTnPk+A5z5PgBb192O0fZTofyha/8AJVDxEuW7vEDcV21XTct16rlVU1UzziqJu1cpifXh4fOfJ8D4AAAAAAAADlboquV00UUzVVVPKIiOczPihaB1LPDSjhlwpwdNyrFNGs5/LN1Or14u1RHK37VFPKn2+lProg9Q3wynevE6ncepY/T0bbs05FXSp503smfyVHd7/KYmufcx41igAAAAAAPP3Jo+n7h0DP0PVcenIwc+xXj5FqqPxqKo5T7/AIp8aqLi3srUOHvEHVtp6jFc14V6Ys3Zp5Rfsz3bdyPJVTMT7fOPWW2ot9X/AMMvt9s3H4gaXjxVqOiU9jzopp9VcxKp/G/+OqeftVVeIECwAAAAAGd9T54c9kefcT0tLBGd9T54c9kefcT0tILXQAAAJV9Xeo24q13aqo1DbHKapn/fbnj/AIawUBXv2mvFX9obY+O3Pq32Oo04q8456jtiI592fsy59WsHAQa0LqJNz3r1H273npGJamfVfYmPcv1RHk6XRj/Nuzhn1KnDDaF6znahjZG5c+3PSi5qUxNmmryWaYin/u6TfIDjat0WrdNu3RTRRTEU000xyiIjvREeJyAAAAmeUTMzyGherU4pUbC4ZXdF07JinX9foqxseKZ9VZsd67d8ncnox5aufrSCF3VO76p4g8Ztb1rGu9k06zc+wsCefcmxa50xVHkqnpVfzNZAAR3ZHb0bT8rVtXw9LwrfZMrMv0Y9mj9auuqKaY+GYBYV1BO1I2/wNtavetdHK17LuZk1T3+xU/g7ce16mqqPdykG8nZ2h4u2tqaVt/DiPsfTsO1i25iOXOKKYp5+/wAufvvWAAAlWJ1X+543Rx/3HetXIrx9Puxp1mY70RZjo1f6+msh33r1ja+zNZ3FkTTFvTcK7lT0u9PQomYj35iI99URqGTezc6/mZNc3L9+5VduVTPdqqqnnMz78g/AABuHqN9yxtrqgtu13K4ox9Srr029z9fs0cqI/wASKGnna0jNv6ZquJqOLV0L+Leov2qonvVUVRVH+cAuOHl7S1ixuDa+l67i1U1WNRw7WVbmmeccrlEVf/t6gAAI69X/ALV+3vBWjXLVuasjQc2jI5xHOew3Pwdce1zmif5VeK3zf+38fdeyda23lRHYtSwbuNMzHPozXTMRV7cTyn3lRWo4mRgZ+Rg5dubeRj3arV2ie/TVTMxMfDEg/AAAAE7utt+Dbc3nin0NCVSKvW2/BtubzxT6GhKoAABE7rk/6DbU853fRJYondcn/QbannO76IEGEw+t48Sps5ufwz1O/wDg70VZuldKe9XH5W1HtxyriPJV40PHrbP1/Utrbo03cWkXuw52nZFGRZq9bpUzz5T44nuxMeKZBcEMd4a7s0/fOxdI3Xpcx9jajjU3ehFXObdferony01RVTPtMiAABgnG7YVjiBtC1p8U2qdQwM6xqGBdr/uXbVcTMc/FVT0qZ9uJ9ZncAAAAACpbjV4YN5efc301bEGX8avDBvLz7m+mrYgAAAAA2V1M+w54icYtF0K9Z7Jp9q59mah3O59j2piaon3U9Gj+ZrVPXreuwp0bh/nb4zbHRy9du9ixZqjuxjWpmOce6r6XtxTHkBKKmIppimIiIiOURHrPoAAA111SG+6eHfB/W9wW7nQz5tfYunx685FznTRMe57tftUyquuV1XK6q66pqqqnnNUzzmZ8cpTdcO37Gr7507YmFe6WLotr7Iy4iY5Tk3Y7kT7m3y9+uUVwAAAAAAAATC62l+dd7/wML595NZCnraX513v/AAML595NYAABHnrgXgBq87Yv9K0hkeeuBeAGrzti/wBKwV2gAPR2z+kWm/8AV2vnw856O2f0i03/AKu18+AXDUfix7T6+Ufix7T6AABKqDqgfDpvr/3DnenrWvyqg6oHw6b6/wDcOd6esGDAAAAAAnH1tn9D92+cLHoqktUSuts/ofu3zhY9FUlqAAAir1yPwa7a88z6CtKpFXrkfg12155n0FYIIgAMu4PXrNjiHpt2/dt2rdMXedddUUxH4Kv15YiPHEWvPWqreunGiY60nBYmcLiLd+I14sxOnynVLH7b6V+1MH4xR9J9t9K/amD8Yo+lE/mc2a5LUbSerxb31h3dhH1eCWH230r9qYPxij6T7b6V+1MH4xR9KJ/M5nJajaT1eJ6w7uwj6vBLD7b6V+1MH4xR9Lt2btq/apu2blF23V+LVRVExPtTCInNJThJ4O9I/h1+kqVea5NTgLUXIr11nTm+bQcHeFNecYiqzVb4uka8+v8AMR7viyoBQtgPzyL9jHtTdyL1uzbiYiarlUUx3fLL9GD8b/B9lfx7PznvhbPn71NuZ01mIRMwxU4TC3L8RrxYmdPlDKvtvpX7UwfjFH0n230r9qYPxij6UT+ZzavktRtJ6vFzr1h3dhH1eCWH230r9qYPxij6T7b6V+1MH4xR9KJ/M5nJajaT1eJ6w7uwj6vBLD7b6V+08H4xR9KOHEq5bvb61e5auUXKKsmZpqpnnE9yO9MMe5vixy3J6cBcmuK9dY05lHn3CivOLNNqq3xdJ159f409wAumVAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAbN6lf+0JsvznT82paXCrTqV/7Qmy/OdPzalpcAAAOvqX5uyf4VfzZdh19S/N2T/Cr+bIKcr/5av3U/wBXBzv/AJav3U/1cAAAAAAAAAAAAAH7YWNfzMyzh4tqq9fv3Kbdq3THOa6qp5REeWZmIfik71AnDKdx76u791LHmrTdAq6OJ0qfU3MyY7n/AGUz0vbmgEuep64eY/DPhdpe3IoonP6H2RqN2nu9kya4ia+768R3KY8lMNhAAAAMa4o7w0/YWwtX3ZqUxNnT8eblNvnym7cnuUW48tVUxHvtLdRHxgz+IGg6voO5cz7I17AyK8uiuqqed3Hu1zPKOfrUVzNPkpmiPWBI4AB+Go4eNqGBkYGbYov4uTaqs3rVceproqiYqpnyTEzD9wFUvH3h9k8M+KGq7ZuRXViU19n0+7V/xcauZmiefrzHdpny0ywFYh1dHDL7suGn3Uabjdk1jblNV/1Mequ4s927T5ejyiuPaq5d9XeAAAAAzvqfPDnsjz7ielpYIzvqfPDnsjz7ielpBa6AAAAAAAAAAAADD+K3EjanDTbdetbn1GixTMTGPjUTFV/Jrj+7bo9efHPej15gHc4k700HYG0M3c+4sqLGHjU+ppj8e9cn8W3RHr1VT3o9+e5EqueL2/tZ4lb7z91a1VFNy/V0LFimedGPZp/Et0+SI78+vMzPrvd4+8Ytx8W9y/ZupVTiaVjVVRp+m26+duxTP96Z/vXJjv1e9HKO41oAAA3b1E21fun4+6RduWpuY2j27mpXu53ImiIpt8//AJK6Gkk5etxbVjE2juHeF63+E1DKpwrFU/8Ap2o6VXL26q4j+UEs47wAAAOjr2kabr2kZOkaxhWM7Ayqehfx71PSouU8+fKY9eO4wn7x3CL93e3PiVLYgDXf3juEX7utufEqT7x3CL93W3PiVLYgDXf3juEX7utufEqT7x3CL93e3PiVLYgDp6Jpen6LpONpOlYdnDwcW3FuxYtU9Gi3THeiI9aHcAAACe8rJ6sja33K8f8AX7Vu30MbU6qdSx+VPKJi7HOvl/8AJFyPeWbIf9ci2t2XSds7zs2p6Vi7Xp2RXEf3a47Jb5+1NNfwghOAAACd3W2/BtubzxT6GhKpFXrbfg23N54p9DQlUAAAid1yf9Btqec7voksUTuuT/oNtTznd9ECDAAJedbz4l/Yer5vDPU8iYs501ZmlzVPcpvUx+Ftx7qmOlEeOmrxpuKedsa1qG3Nw6fr2lXps52BkUZFivxV0zzjn5PWnyLYOF28dN37sLSd16XVHYM/Hiuq3z5zZuR3K7c+WmqJj3gZMAAAAAAACpbjV4YN5efc301bEGX8avDBvLz7m+mrYgAAAAD3Nhbczt37z0jbOm0TVlall0Y9E+tT0p7tU+SmOdU+SJW1bX0XB25tzTtB023FvD0/Gt41inxUUUxEe/3OaF3W69hfZ+5NW4gZtmJsabR9hYM1R379cc7lUe5o5R/8nkTjAAAePvfcWBtPaOq7k1Ovo4mm4teRc7vdqimOcUx5ZnlEeWYewif1xHf32u2npewMG/yyNUufZedET3Yx7c+opnyVV93/AOPyghXu3Xc/c25tS3BqlybmbqGTcyb08/71U8+UeSO9Hkh5YAAAAAAAAAmF1tL8673/AIGF8+8mshT1tL8673/gYXz7yawAACPPXAvADV52xf6VpDI89cC8ANXnbF/pWCu0AB6O2f0i03/q7Xz4ec7+3a6bev6fXXVFNNOVamZn1o6cAuIo/Fj2n1xtzE0UzHemIcgAAJVQdUD4dN9f+4c709a1+VU3VGWKsfjzvm3Xz5zruVc7scu5VcmqP8pBgAAAAAAJx9bZ/Q/dvnCx6KpLVE7rbdmqNi7qyJ59GvVLVEdz9Wz3e7/MliAAAir1yPwa7a88z6CtKpFLrkdyI4fbYscu7Vq1dfPn+rZn6QQUAAAAAAAASV4S+DvSP4dfpKkakleEvg70j+HX6Spm+E/Rad79S3fk/wCn3NyfypZUAwzrYwfjf4Psr+PZ+czhg/G/wfZX8ez85Oy3pdrejtVOfezL+5V2I8AOnvn8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABs3qV/wC0JsvznT82paXCrTqV/wC0JsvznT82paXAAADr6l+bsn+FX82XYdfUvzdk/wAKv5sgpyv/AJav3U/1cHO/+Wr91P8AVwAAAAAAAAAAAAB3dB0rO1zWsLR9Mx6sjNzb9Fixap79ddUxER8MrW+DexsLhzw50naeF0a5xLMTk3af+Nfq7tyv36ufLyREImdb34ZRqOv5nErVcfnjabM4ulxVHcrv1R+EuR7imYiPLXPiTjAAABi/FfeWBsDh9rG7NQmmbeBjzXbtzPLs12e5btx7qqYgEReuFcTPtjr2Fw10u/P2Pps05epzRV3K79VP4O3PL9WmelPlrjxI+cFd9ZnDniVpG68Wa5t4t7o5Vqmfy1iruXKP+3ux5YifWY3uLV8/Xteztb1TIqyM7OyK8jIuVf3q65mZn4ZdAFx2k5+Jqul4mp4F+jIxMuzRfsXaJ5xXRVEVU1R7cTDtIt9b94lU65srJ2BqWTE6hon4XCiqe7cxKp70ePoVTy9qqlKQAAHG7bou26rdyimuiqJiqmqOcTE9+JjxKveqh4a18MuK2fpWPamnR8z/AGzS6+Xc7DVPdo9uirnT7URPrrRGkurI4ZRxD4U5GTgY3ZNc0PpZuF0aeddyiI/C2o91THOI/WppBWoAAAAzvqfPDnsjz7ielpYIzvqfPDnsjz7ielpBa6AAAACINzq4dMorqp+91mTymY/OtP1YJfCIHbx6X+7rM+Vafqzt49L/AHdZnyrT9WCX4h5f6uTBi3zs8N8muvn3q9Xppj4YtSx3V+rd3PeiY0nZGkYk+tOTlXL/AC/7YoBOV5G6dz7e2tp9Wobj1rA0rFj/AIuVfptxPkjnPOZ8kK7N19VLxi12iu1a3BZ0ezVz50abi02p5e7npVR7cTDUGt61q+uZtWdrWqZupZVf417Lv1Xa5/mqmZBNLjD1ZOkYNF/TeGunTqeTymmNTzaKrdiifHRb7lVf83RjyShzvXdm4t567e1zc+rZOp593v3b1X4setTTTHcppj1qYiIeIAAAAA+xEzPKO+tb6n3asbM4NbY2/VRFN+xg0XMn+Nc/CXP9Vcx7UQrg6nvas704zbY2/VbmuxezqLuTH/8ATa/CXPhppmPbmFrcRyjlAAAAAAAAAAAAAAADWvVPbVjePAzc+k02unk28SczG8cXbM9kjl7fRmn2plspxuUU3KKqK6YqpqjlVExziY9eAU1T33xlvGTbFezOKe5Ns1RyowdQuUWfLamelbn36KqZYkAACd3W2/BtubzxT6GhKpFXrbfg23N54p9DQlUAAAid1yf9Btqec7voksUTuuT/AKDbU853fRAgwAAlr1vTiVOnbhzuG+p5HLG1Lnl6Z06u5TkU0/hKI5/rURz9uifGiU72garn6FrmDrOl5FWPnYN+jIx7tPforomJifhgFxQxXhLvTB4g8PdI3ZgdGmjOsRVdtxPPsN2O5ctz7mqJj2uTKgAAAAAAVLcavDBvLz7m+mrYgy/jV4YN5efc301bEAAAH6Y9m7kX7dixbquXblUUUUUxzmqqZ5RER45l+be3URbC+7PjRiahl2YuaZt+mNQv84iYquxPKzR/3+q9qiQTo4C7Ht8POFOh7Y6NMZVnHi7m1R/eyK/VXJ8vKZ6MeSmGdEAAAOF+7bs2a7t2umi3RTNVVVU8opiO7Myqo6oHfNziHxa1zcsVzOJcv9hwaZnuU49v1Nv4YjpT5apTp6tffs7L4K5uHiX+x6nr9X2ux+jPKqmiqOd6uPao508/WmuFbAAAAAAAAAAAJhdbS/Ou9/4GF8+8mshT1tL8673/AIGF8+8msAAAjz1wLwA1edsX+laQyPPXAvADV52xf6VgrtAAfaZmmqKqZmJjuxMes+Ed8Fwm0NRo1faekatbmJozcGxkU8vFXbpqj+r1GpOpA3BG4ep82xem5Nd7CsVYF3n602appiP+3ottgAAK2erj2/c0TqhNXyuhNNjVrNjOtT0eUTzoiir/AFUVT76yZGvq9OGd7dnD/H3fpGNVe1Pb3Sqv0UU86ruJV3a/bmiYir2umCvsAAAAGWcJNjarxF37pu1dJonsmVc5373LnTj2Y/HuVeSI+GeUeuCeHUG7eu6LwBw82/b6FzWM2/nREx3ehzi3R70xb5x7pvx0Nu6Rg6DoOBommWYs4WBj0Y2PRH92iimKYj4Id8AABDPrlep93Zej0zHP/a8mqPXj8nTH/wDr4EzFdPV8bip1vjze0+1cpqtaLgWcP1M9zpzzu1e/zuRHvAj6AAAAAAAAkrwl8Hekfw6/SVI1JK8JfB3pH8Ov0lTN8J+i0736lu/J/wBPubk/lSyoBhnWxg/G/wAH2V/Hs/OZwwfjf4Psr+PZ+cnZb0u1vR2qnPvZl/cq7EeAHT3z+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA2b1K/8AaE2X5zp+bUtLhVp1K/8AaE2X5zp+bUtLgAAB+GoU1V4ORRRTNVVVqqIiPXnlL9wFV17gdxdm7XMcO9x8pqn/AOzq8bj943i7+7vcfxOpal0af1Y+A6NP6sfACq37xvF393e4/idR943i7+7vcfxOpal0af1Y+A6NP6sfACq37xvF393e4/idR943i7+7vcfxOpal0af1Y+A6NP6sfACq37xvF393e4/idR943i7+7vcfxOpal0af1Y+A6NP6sfACq37xvF393e4/idTwt48Pd7bOw7GZuja+qaRj37k27VzLsTRTXVy58omfX5d1bf0af1Y+BFLrkkRHD/a3KIj/AOq3PQyCCoAD19mbe1Ldm69M23pFmbudqOTRj2aeXciap/GnxREc5mfWiJeQmf1vPhl0aM3ifqtju1dLC0iKvF3r12PmRPuwSo4b7T03Y2x9J2rpVERi6dj02oq5cpuVd+uufLVVM1T7bIQAAAQX64PxM+2u5sThvpl/pYmkzGTqM0VdyvJqp9RRPuKJ5+3X5EveLu9MHh9w71jdmd0aowceZs2pnl2a9PqbdHv1TEe1zlVBr2q52ua1m6xqeRVkZubfryMi7V3666pmZn4ZB0QAZfwc3vm8O+I+j7sw+lVGHfj7ItRPLs1iruXKPfpmeXl5T6y13RNSwtZ0jD1bTr9ORh5lii/Yu0zziuiqImmfglTmnf1vriXGs7RyuHepZPSztGib+B06u7Xi1Veqpjx9CufgrjxAlSAASAK1OrH4ZTw84r5GRgY0WtD1zpZuD0I5U26pn8LajxdGqecR+rVS0ktE6qPhrRxN4UZ+l41qKtZwv9s0ur1+zUx3aParp50+3MT6yr25RXbuVW7lNVFdMzFVNUcpiY78TAOIADO+p88OeyPPuJ6WlgjO+p88OeyPPuJ6WkFroAAAE95TXkfl7nup/quUnvKa8j8vc91P9QfmAAAAAAAAAABHfBLTrcO1fsvdu4d4XrXO3gYtGFYqn/1Ls9Krl7VNER/MnI0n1FG1fuX4BaRcuWux5OsV16ne7nKZ7Jyi3/8AjpobsAAAYNxp4n7f4UbWsbg3DZzL9m/l04tqziU01XKq5pqq58qpiOURTMz3WcoNdce3TOXu7bm0bN3nb0/Erzb9Mf8AqXZ6NPP2qaOf80g2T26fDT9g7p/wLP1h26fDT9g7p/wLP1iAYCfnbp8NP2Dun/As/WHbp8NP2Dun/As/WIBgJ+dunw0/YO6f8Cz9Ydunw0/YO6f8Cz9YgGAn526fDT9g7p/wLP1jdHBviRoXFLZ/3TaBay7ONGTcxq7WVTTFyiujlPdimZjuxVEx3e9KptMXrbe5+x526dn3bkdG7Ra1HHomr16Z7Hc5R5Ym38AJpgAAAgR1xTa06ZxM0jdFq1NNnWcHsdyqI7k3rE8p/wBFdv4EXljHV57W+3/AvI1S3b6WRoWXbzaZjv8AY5nsdyPa5VxP8qucAAE7utt+Dbc3nin0NCVSKvW2/BtubzxT6GhKoAABE7rk/wCg21POd30SWKJ3XJ/0G2p5zu+iBBgAAAEq+t88Sp0fdmVw61PJ6ODrHPI0/p1epoyqafVUx7uiPhojxp2qc9D1PN0XWMPV9Nv1Y+bhX6L9i7T36K6ZiaZ+GFrnBnfGHxF4b6RuvEimivLsxGTaiefYb9PcuUe9VE8vJMAzAAAAAAFS3Grwwby8+5vpq2IMv41eGDeXn3N9NWxAAACFkXURbCnZnBjF1HMsTb1PcFcahf6UTFVNqY5WaP8As9V7dcoL8Btj3eIfFbQ9sRFX2Lfvxczao/uY9Hqrk+/Eco8tULWsezax7FuxYt027VumKKKKY5RTTEcoiI8UQD9AAAAQd6s7bPFPiJxV7Ho2yNwZehaNYjFw7tvEmbd6urlVdu0z68TPKnn4qIaO+8bxd/d3uP4nUtS6NP6sfAdGn9WPgBVb943i7+7vcfxOo+8bxd/d3uP4nUtS6NP6sfAdGn9WPgBVb943i7+7vcfxOo+8bxd/d3uP4nUtS6NP6sfAdGn9WPgBVb943i7+7vcfxOo+8bxd/d3uP4nUtS6NP6sfAdGn9WPgBVZVwP4u00zVPDvccREc5/2OpryqJpqmmqOUxPKYXH59NP2Df9TH5Or1vJKnPJ/3i57uf6g/MAEwutpfnXe/8DC+feTWQp62l+dd7/wML595NYAABHnrgXgBq87Yv9K0hkeeuBeAGrzti/0rBXaAAACZHW4N502724th5VyIm50dTw4mr145W7sR73Y596U0FSfCLeWZw/4jaLuzD6VU4GTFV63E8uy2Z9Tco9+mZj2+U+ste0DVcDXdEwtZ0vIpyMHOsUZGPdpnuV0VxE0z8Eg7wAD5XTTXRNFdMVU1RymJjnEw+gIWdUj1J+b9nZe6OF1ii9Zu1VXcjROcU1W5nuzNiZ7k0/8AJPKY/u8+5ERH1nSdT0XULmn6vp+VgZdqeVdjJtVW66fbpqiJXFvJ3Ftnbu48eMfcGg6Zq1qO9Rm4tF6I9rpRPIFP0RM+tL5yWi5XU88GMm9N25sDSqap9a3Ny3HwU1xD19u8G+Fm371N7S9haDau08ujcuYkXa6Z8cVXOlMT5QVz8KODO/8AiVmWqdv6JeowKqoi5qWVTNrFtx689OY9V7VMTKwbqfuDW3eEe25xMDlm6xlRE6hqVdHRrvTHeppj+7bj1qffnnLZlu3RboiiimmmmmOUREcoj2nIAAAAHS17U8PRdEztY1C7FrDwsevIv1/q0UUzVVPwQqO3zuDK3VvHV9yZvPs+pZl3KqiZ59Hp1TMU+9HKPeTf64BxIt6DsKxsPT78fbLXZivKime7bxKJ5zz93VER7VNSAwAAAAAAAADPdtcT9W0LQ8bScfAwrtrHpmmmqvpdKeczPd5T5WBDwxGFtYmni3adYTMFmGJwNc3MPXxZmNP6bO+/Lrn7L0//AF/Sffl1z9l6f/r+lrEQ/Q2B2cLPlRm23n7dzZ335dc/Zen/AOv6Xk7t4kanuPRLmlZWDh2rdyumua7fS6UdGefrywcfu3lWDt1RXTbiJh5XuEWZ37dVu5emaZjSY/45uoAWClAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAbN6lf+0JsvznT82paXCrTqV/7Qmy/OdPzalpcAAAAAAAAAAAAAIo9cl8H+1vOtz0MpXIo9cl8H+1vOtz0MggqADJOGe0NS33vrSdqaVTP2RqGRFua4p5xao79dyfJTTEz7y1/aOg6btfbGnbe0ixFnB0/Hox7FHLu9GmOXOfHM92Zn15mUY+t88MvtVtzL4k6rjcsvVYnG0zpx3aMamr1dcR63TqjlHko8UpYgAAAxDjHvfC4d8ONY3ZmzRM4dmfse1VP5a/V6m3RHt1THPyc59YEQeuC8S6tY3Zi8O9NyInB0eYv5/Rn8fKqp9TTPuKJ+GufEio7mtalmaxrGZq2o36sjMzL9d+/dqnnNdddU1VTPtzMumAAAyzhFvXN4e8RNH3Zg9KqcK/E3rVM/lrM9y5R79MzHt8mJgLi9C1TB1vRcLWNMyKcjCzbFGRj3ae9XRXETTPwS7qKHW+OJdOq7Yy+HGp3+ebpUVZOndKru141VXq6I9xXPP2q/IleAAAru6ufhl9xnEydz6bj9j0fcc1X4imPU2sqPytHkirnFce6q5d5YiwHj/wAPcbiZwv1TbNyKKcyqjs+n3ao/JZNETNE8/Wie7TPkqkFUo/fUMPJ0/PyMDNsV2MrGu1Wb1quOVVFdMzFVM+WJiYfgAzvqfPDnsjz7ielpYIzvqfPDnsjz7ielpBa6AAABPeU15H5e57qf6rlJ7ymvI/L3PdT/AFB+YAAAAAAAAAD19maHlbm3ZpO3sOKpv6lmWsWjlHPlNdUU8/e58/eeQkJ1A+1Pt/xxt6vetRVi6Dh3MuZmO52Wr8Hbj2/VVVR7gFhWj4GNpWk4el4VHY8bDsUY9mn9WiimKaY+CIdojuRyAAAJVUdUZun7seNe6dct3JuY9edXYxp584mza/B0THtxTz99ZJxw3PGzuEm5tx9OKLmHp9zsEzPL8LXHQtx/31UqnK6pqrmqqZmZnnMzPfkHEAAAAABtnqRtz/crx/2zlXLkUY2bfnT8jnPKJpvR0Kec+KK+hPvNTP3wMm9hZtjMx65ovWLlNy3VHrVUzzifhgFyMDxNha7Z3PsrRdxY9UTb1LBs5UeTp0RVMe9My9sAAHlbw0XG3HtTVtAzKYnH1LDu4tzyRcomnn7fd5qh9Z0/J0nV8zS8yjoZOHfrsXqfFXRVNNUfDErjZ7sK0+rY2rO2OP2sXbduacXWKKNSsz60zcjlc/8AyU1/DANJgAnd1tvwbbm88U+hoSqRV6234NtzeeKfQ0JVAAAIndcn/QbannO76JLFE7rk/wCg21POd30QIMAAAAJSdb94l/aHeeTw/wBTv8sDXJ7LhdLvW8umnveTp0Ry9umnxotuzpWdlaZqeLqWDersZeLeov2LtE8qqK6ZiqmqPLExEguPGFcEN+YfEjhppO6sWqiLuRa6GZapn8jkU9y5R8PdjyTE+uzUAAAAFS3Grwwby8+5vpq2IMv41eGDeXn3N9NWxAAHp7V0TO3JuTTtA0y3NzM1DJt41mn/AJq6oiPejnz94E0Ot17C+wNt6txAzrPK/qVf2FgTMd6xbnncqj3VfKP/AI/Klq8XYm3MHaOzdI2zptEU4um4lvHonl3aujHdqny1TzqnyzL2gAAAAAAAAAAAAfhqH+43/wCHV/SVOWV/vNz3c/1XG6h/uN/+HV/SVOWV/vNz3c/1B+YAJhdbS/Ou9/4GF8+8mshT1tL8673/AIGF8+8msAAAjz1wLwA1edsX+laQyPPXAvADV52xf6VgrtAAAATJ6gbjFbtR96vcOV0Yqqqu6HduV9znPOa8fu+OedVPl6UeKENn64mRfxMq1lYt65Yv2a6blq5bqmmqiqJ5xVEx3YmJjnzBcmI+9Sdx/wALiRpFnbW48i3j7uxLXKrpcqadQoj/AIlH/Py/Gp9+O5z5SCAAAAAAAAAAAeBxB3bo2xtoahufXsmmxg4Nqa6u76q5V3qaKY9eqqeURHlejuDWNL0DRsrWNazrGBp+Jbm5fyL1XRoopj15n+kd+Z7kK3+qm445vFncdOHp3ZcTa2n3J+wceruVX6u9N+5H60+tH92J8cyDX3Fbe2q8Q9+anuzV6uV7Nu87dmKpmmxajuUW6fJTTyjyzzn12LAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADZvUr/wBoTZfnOn5tS0uFUXU/a7pO2eMu19e1zLjD03CzqbuRfmiqqKKYpq7vKmJme/60J+9s1wO9ndn5PyvqgbgGn+2a4Hezuz8n5X1R2zXA72d2fk/K+qBuAaf7Zrgd7O7PyflfVHbNcDvZ3Z+T8r6oG4Bp/tmuB3s7s/J+V9Uds1wO9ndn5PyvqgbgGn+2a4Hezuz8n5X1R2zXA72d2fk/K+qBuAaf7Zrgd7O7PyflfVHbNcDvZ3Z+T8r6oG4Bp/tmuB3s7s/J+V9Uds1wO9ndn5PyvqgbgRR65L4P9redbnoZbO7Zrgd7O7PyflfVI9dXBxY4fcQ9naBg7O3Fb1TJxdQru3qKca9b6FE25iJ510UxPd8QImM04KbEzeI/EnSdqYnSot5N3p5d6mOfYcenu3K/ejuR5ZiGFpYdRZvTg/w229qOt7r3Zj4m5dTr7D2KcPIrnHxqJ7lPSotzTzqq9VPKe9FPikE3dF03C0bSMTSdOsUY+Hh2aLFi1THKKKKYiKYj3odtp/tmuB3s7s/J+V9Uds1wO9ndn5PyvqgbgGn+2a4Hezuz8n5X1R2zXA72d2fk/K+qBuBA3rgXEqrXN543D7TMiZ0/ReV3O6NXcuZdUdyJ9xRPL26qvE35vHqpuEmBtbUszQN0WtU1W1jV1YeJGFkU9mu8vUUzNVuIiOfLnznvRKufV9QzNV1TK1PUL9WRmZd6u/fu1d+uuqZmqZ9uZkHVAAAAABlPCfeeocP+IOj7s06apuYGRFV21E8ovWp7ly3PuqZmPb5T6y2Hb2rYGvaFg61pd+m/g52PRkY9yme5VRXTExPwSp1TB6jbqgtsbT2Lk7P4ga39rrGBd7Jpd+uxduxVbrmZrteopqmOjVzmOfrV8vWBNcaf7Zrgd7O7PyflfVHbNcDvZ3Z+T8r6oG4Bp/tmuB3s7s/J+V9Uds1wO9ndn5PyvqgRq6v7hj9oN6WN/wCl48xp2uVdDNimn1NrLpjvz/Epjn7dNXjRcWG8XOMnU/cQ+HurbT1DfWNFObZmLN2rTsr8Deju27kfgvWqiJ9rnHrq9b1MUXa6KblNyKapiK6efKryxz5TyBwZ31Pnhz2R59xPS0sEZbwa1bT9B4sbV1rVsmMbAwdWx8jJvTTNUW7dNyJqq5UxMzyiPWgFtY0/2zXA72d2fk/K+qO2a4Hezuz8n5X1QNwDT/bNcDvZ3Z+T8r6o7Zrgd7O7PyflfVA3BPeU15H5e57qf6rNJ6prgdy/Tuz8n5X1Ssq9MVXq6onnE1TMfCDgAAAAAAAAAAn31u/asaXwr1LdF230b2t50026pp5c7Nn1Ed33c3PgQFjvrEOFHHTgXszhtt/a9G+rETp2Bas3OWn5Xducudc9y169c1T74JDjT/bNcDvZ3Z+T8r6o7Zrgd7O7PyflfVA3ANP9s1wO9ndn5PyvqjtmuB3s7s/J+V9UDXXXFt0fa/hvo+1bVyIu6vndmu08+7NqxHPveLp1Uf8AagY3d1ZfEnSeJHFSzlbc1H7O0PT8C3j412LddFNdczNdyqKa4iY7tUU92P7kNIgAAAAAAAAsW6gnc3284FWNKu3enf0PMu4kxPfi3VPZaPnzHvJBK9Ooe4sbc4c7g3Dhbw1b7W6TqOLbuW7tVq5ciL9urlFPRopqmOdNdXd5f3YSr7Zrgd7O7PyflfVA3ANP9s1wO9ndn5PyvqjtmuB3s7s/J+V9UDcCJPXH9qzk7V27vGzb51YOVXg35j9S7HSomfJFVEx/M2t2zXA72d2fk/K+qYJ1QHGngnv3g/uLbGNvbGu5mTizXh01YOVTzv25iu3HObXKOdVMR74IBD7PffATu6234NtzeeKfQ0JVIO9RBxa4e8Pdj67p+8dx29LysrUovWbdWNeudKjsVNPPnRRVEd2J76QXbNcDvZ3Z+T8r6oG4Bp/tmuB3s7s/J+V9Uds1wO9ndn5PyvqgbgRO65P+g21POd30TaPbNcDvZ3Z+T8r6pHfq4uK/D/iHtPb2Fs7cVvVcjEz7l2/RTjXrfQom3yiedyimJ7viBE8AAAAAEnOoB4kfc7v+/sbUb/R07X+U4vSnuW8umPUx/PTzp9uKU/FN2Bl5ODnWM3DvV2MnHuU3bN2ieVVFdM86aonxxMRKxXYHVTcK8/Zmk5e59z2tL1uvGp+z8WcPIr7HejuVcpptzExMxzjlM9yY9cG/Bp/tmuB3s7s/J+V9Uds1wO9ndn5PyvqgbgGn+2a4Hezuz8n5X1R2zXA72d2fk/K+qBXtxq8MG8vPub6atiDJOKOo4WscStzatpt+MjCzNWysjHuxTNPTt13aqqauUxExziY78c2NgJS9b02DGsb8z99ZtnpYuiW+wYk1R3JybsTEzHubfS9+uEWoT06nPi5wN4bcJNH23e3vjUah0JydRmnByp55NzlNcc4td3o+po5+KmASkGn+2a4Hezuz8n5X1R2zXA72d2fk/K+qBuAaf7Zrgd7O7PyflfVHbNcDvZ3Z+T8r6oG4Bp/tmuB3s7s/J+V9Uds1wO9ndn5PyvqgbgGn+2a4Hezuz8n5X1R2zXA72d2fk/K+qBuAaf7Zrgd7O7PyflfVHbNcDvZ3Z+T8r6oG4Bp/tmuB3s7s/J+V9Uds1wO9ndn5PyvqgbgGn+2a4Hezuz8n5X1R2zXA72d2fk/K+qBtnUP9xv8A8Or+kqcsr/ebnu5/qsvzOqX4IXMW7RTvuzNVVFUR/sGV3+X8JWffqiq/XVTPOJqmY+EHAAEwutpfnXe/8DC+feTWV99QzxJ2Tw71Ddd3eWuUaVRnWcWnGmqxdudkmiq7NX5Omrly6Ud/xpSds1wO9ndn5PyvqgbgGn+2a4Hezuz8n5X1R2zXA72d2fk/K+qBuBHnrgXgBq87Yv8AStk3bNcDvZ3Z+T8r6ppnqxuM/DPffB6rQtqbot6lqM6jYvdhpxb9uehT0+lPOuiI7nOPXBC0AAAAAH74GZl6fm2c3Byb2LlWK4uWr1muaK7dUTziqmY7sTHjTW6nXqtMLOs422+KN2jEzIiLdnWop5Wr3rR2aI/Eq/54joz68U+vCIBcnh5ONmYtrKxL9rIsXqIrt3bVcVUV0z3piY7kx5Yfqqr4T8aOIPDO5TRtzWq6tP6XSr07KjsuNV3ec+omfUzPjpmJSt4ddWdtDU7drH3ro2boWT3IryMWPsnHmfHyjlXTHk5Ve3IJTjENp8T+Hm67dNe395aLnVVRz7FTlU03Y9u3Vyqj34ZbRXTXTFVM9Kme7Ex3YkHIOceM5wAOnqWq6Zplmb2pahiYVuI5zXkXqbcRHj51TDVO9+qW4QbXorondFrWMmnnysaVRORMz7uPUR79QNxMG4tcVtl8MdInN3PqlFu/VRNWPg2ZivJyJ8VFHi/5p5Ux40ReKfVkbt1qi5g7H0uzt3Fq7n2XfmL+VMeOP7lHwVT4pRn1vVdT1vU7+p6vn5Ofm5FXSu5GRdm5XXPlme6DZnVA8dN08WtTm1k1Tpu37NfSxdLtXJmmJ9au5Pc6dfl70etEd2Z1MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPsTMPZ0vdm6dK5fazcesYXLvfY+bco/pU8UBnFni/wAVLNM02uIu66YmefKNVvf+Tq5/E7iNn01UZm/NzX6ap5zFeqXpif8AUxEB2c7UM7OudkzczIyq/wBa9dqrn/OZdYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAf/Z"

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
        f'<img src="data:image/jpeg;base64,{LOGO_BASE64}" alt="Expanzio+"/>'
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
