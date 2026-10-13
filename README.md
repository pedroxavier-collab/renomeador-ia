# RenomIAdor · Expanzio+

Renomeia arquivos no padrão de nomenclatura da Expanzio+ usando IA (Claude, da Anthropic, ou Gemini, do Google).

**Padrão:** `CLIENTE-NOME ABREVIADO.ext`
Quando o documento tem data de emissão e de vencimento: `CLIENTE-NOME ABREVIADO-EMISSÃO-VENCIMENTO.ext`

**Exemplos:**
`DIA-NOT. EXIG. PUBLI. 29-23.pdf`
`DIA-LIC. FUNC-10.01.2026-10.01.2027.pdf`

## Como usar

1. Escolha o cliente na lista. Dá para digitar para pesquisar; se o cliente não existir, digite o nome, aperte Enter e clique em **Adicionar à lista**.
2. Selecione os arquivos (PDF, TXT, PNG, JPG).
3. Clique em **Gerar sugestões de nome**. A IA lê cada arquivo e sugere o nome abreviado e, se houver, as datas de emissão e vencimento.
4. Revise a tabela: corrija nomes e datas e preencha os arquivos que a IA não conseguiu ler.
5. Marque a confirmação e baixe o `.zip` com os arquivos renomeados.

O download só libera quando todos os nomes estão preenchidos, nenhum arquivo fica com nome igual a outro, nenhum vencimento é anterior à emissão e a pessoa confirma que conferiu.

As datas só entram no nome quando emissão e vencimento estão preenchidas. O formato das datas é definido em `FORMATO_DATA_NOME` no `app.py`.

## Estrutura

```
app.py                  aplicativo
requirements.txt        bibliotecas usadas
.streamlit/config.toml  tema visual (cores e fontes da marca)
assets/                 logo e símbolo oficiais da Expanzio+
clientes.csv            lista de clientes (colunas Cliente e Sigla)
```

## Configuração (Secrets do Streamlit)

No painel do Streamlit Cloud: app > Settings > Secrets.

```toml
# Qual IA usar: "claude" (padrão) ou "gemini"
IA_PROVEDOR = "claude"

# Claude
ANTHROPIC_API_KEY = "sua-chave"
# Opcional. Padrão: claude-haiku-5-5 (mais rápido e barato).
# Se a leitura errar muito, teste: claude-sonnet-5-5
# CLAUDE_MODEL = "claude-sonnet-5-5"
# Só se a chave não tiver sido criada dentro de um workspace:
# ANTHROPIC_WORKSPACE_ID = "wrkspc_..."

# Gemini (só precisa se IA_PROVEDOR = "gemini")
GEMINI_API_KEY = "sua-chave"
# GEMINI_MODEL = "gemini-2.5-flash"
```

Trocar de IA ou de modelo é só mudar os Secrets. Não precisa mexer no código.

## Lista de clientes

A lista fica em `clientes.csv`. A coluna **Sigla** é opcional: quando preenchida, é ela que vai no nome do arquivo (ex: Smart Fit → `SMART`); sem sigla, vai o nome do cliente. Para incluir ou corrigir siglas em massa, edite o arquivo direto no GitHub.

Clientes adicionados pelo app só ficam salvos de vez se o app tiver permissão para gravar no repositório. Para isso, acrescente aos Secrets:

```toml
GITHUB_TOKEN = "token-do-github"
GITHUB_REPO = "usuario/nome-do-repositorio"
# GITHUB_BRANCH = "main"   # só se o ramo principal tiver outro nome
```

O token é criado no GitHub em Settings > Developer settings > Personal access tokens > Fine-grained tokens, com acesso só a este repositório e permissão **Contents: Read and write**. Sem essa configuração, o cliente adicionado vale até o app reiniciar.

A lista contém nomes de clientes, inclusive pessoas físicas: mantenha o repositório **privado**.

## Rodar no computador

```bash
pip install -r requirements.txt
set ANTHROPIC_API_KEY=sua-chave        # Windows
export ANTHROPIC_API_KEY=sua-chave     # Mac/Linux
streamlit run app.py
```

## Dados enviados à IA

Cada arquivo é enviado à IA escolhida para leitura, junto da requisição. No Gemini, arquivos acima de 15 MB passam por um upload temporário que o app apaga logo em seguida. No Claude, arquivos acima de 20 MB não são enviados e o nome é preenchido à mão. Nenhum arquivo fica gravado no servidor do app.

O uso com documentos de clientes depende de homologação interna: {A CONFIRMAR}.

## Manutenção

Responsável: {A CONFIRMAR}
