FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Baixa o código oficial do OpenManus
RUN git clone --depth 1 https://github.com/FoundationAgents/OpenManus.git /app/openmanus

WORKDIR /app/openmanus

# Remove a dependência de navegador automatizado (browser-use/crawl4ai) do
# código-fonte: essas libs puxam PyTorch inteiro (~1.5GB) e não são
# necessárias pro motor de ações do Jarvis (python/bash/arquivos/busca).
RUN sed -i \
    -e '/browser_use_tool import BrowserUseTool/d' \
    -e '/crawl4ai import Crawl4aiTool/d' \
    -e '/"BrowserUseTool",/d' \
    -e '/"Crawl4aiTool",/d' \
    app/tool/__init__.py \
 && sed -i \
    -e '/from app.agent.browser import BrowserAgent/d' \
    -e '/"BrowserAgent",/d' \
    app/agent/__init__.py

# requirements.txt enxuto (sem browser-use, browsergym, gymnasium, crawl4ai, playwright)
COPY requirements.txt /app/openmanus/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

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
