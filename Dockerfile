FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Baixa o código oficial do OpenManus
RUN git clone --depth 1 https://github.com/FoundationAgents/OpenManus.git /app/openmanus

WORKDIR /app/openmanus

# Navegação real do Jarvis (browse_website) precisa de BrowserUseTool/BrowserAgent
# e de um Chromium instalado — mantemos o código-fonte original do OpenManus intacto
# (sem o sed que os removia) e instalamos o browser via Playwright abaixo.
#
# app/agent/browser.py importa SandboxBrowserTool só como fallback de nome de
# ferramenta (nunca instanciada — o BrowserAgent usa só BrowserUseTool()), mas essa
# importação arrasta o pacote `daytona` (sandboxing cloud, não usado neste fork e
# nunca esteve no requirements.txt). Removemos só essa importação e seu fallback,
# mantendo BrowserAgent/BrowserUseTool 100% funcionais.
RUN sed -i \
    -e '/from app.tool.sandbox.sb_browser_tool import SandboxBrowserTool/d' \
    -e '/^        if not browser_tool:$/,/^            )$/d' \
    app/agent/browser.py

COPY requirements.txt /app/openmanus/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install --with-deps chromium

# Pré-baixa o arquivo de tokenização (evita depender disso em tempo de execução)
ENV TIKTOKEN_CACHE_DIR=/app/openmanus/.tiktoken_cache
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Nosso wrapper: expõe um agente leve (sem navegador) como API HTTP
COPY server_wrapper.py /app/openmanus/server_wrapper.py

ENV PYTHONUNBUFFERED=1
EXPOSE 7860

RUN useradd -m -u 1000 user && chown -R user:user /app/openmanus
USER user

CMD ["uvicorn", "server_wrapper:app", "--host", "0.0.0.0", "--port", "7860"]
