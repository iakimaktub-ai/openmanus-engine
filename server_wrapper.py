import os
 
# --- gera config/config.toml a partir de variáveis de ambiente (Secrets do Space) ---
# precisa acontecer ANTES de importar qualquer coisa de app.* (o OpenManus lê o
# config.toml no momento em que o pacote app.config é importado).
 
def _write_config():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    model = os.environ.get("OPENMANUS_MODEL", "gemini-3-flash-preview")
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    os.makedirs("config", exist_ok=True)
    toml_content = (
        "[llm]\n"
        f'model = "{model}"\n'
        f'base_url = "{base_url}"\n'
        f'api_key = "{api_key}"\n'
        "max_tokens = 8192\n"
        "temperature = 0.3\n\n"
        "[llm.vision]\n"
        f'model = "{model}"\n'
        f'base_url = "{base_url}"\n'
        f'api_key = "{api_key}"\n'
        "max_tokens = 8192\n"
        "temperature = 0.3\n\n"
        "[mcp]\n"
        'server_reference = "app.mcp.server"\n\n'
        "[daytona]\n"
        'daytona_api_key = "unused"\n'
    )
    with open("config/config.toml", "w") as f:
        f.write(toml_content)
 
_write_config()
 
# --- agora sim, o resto dos imports ---
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
 
from app.agent.toolcall import ToolCallAgent
from app.tool import (
    Bash,
    PlanningTool,
    StrReplaceEditor,
    Terminate,
    ToolCollection,
    WebSearch,
)
from app.tool.python_execute import PythonExecute
from app.logger import logger
 
 
class JarvisEngine(ToolCallAgent):
    """Agente leve (sem navegador automatizado) que serve de motor de ações pro Jarvis."""
 
    name: str = "JarvisEngine"
    description: str = "Motor de execução autônoma de tarefas do Jarvis"
 
    system_prompt: str = (
        "Você é o motor de execução de tarefas do Jarvis, um assistente pessoal em português do Brasil. "
        "Resolva a tarefa dada de forma autônoma e completa, usando as ferramentas disponíveis: "
        "executar Python, executar comandos bash, ler/editar arquivos, buscar na web e planejar passos. "
        "Seja direto e eficiente. Ao concluir, chame a ferramenta terminate com um resumo claro do "
        "resultado final em português — esse resumo é o que será mostrado à pessoa."
    )
    next_step_prompt: str = (
        "Escolha a ferramenta mais adequada para avançar a tarefa. "
        "Quando a tarefa estiver completa, chame terminate com o resumo do resultado."
    )
 
    max_steps: int = 8
 
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            Bash(),
            StrReplaceEditor(),
            WebSearch(),
            PlanningTool(),
            Terminate(),
        )
    )
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
 
 
app = FastAPI(title="OpenManus Engine for Jarvis")
 
# Troque pelo endereço exato do seu Jarvis no GitHub Pages
ALLOWED_ORIGINS = [
    "https://iakimaktub-ai.github.io",
]
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)
 
# Token simples pra ninguém além do seu Jarvis conseguir usar sua chave/quota.
# Configure o mesmo valor como Secret WRAPPER_AUTH_TOKEN aqui no Space, e
# cole esse mesmo valor no campo de token do Jarvis.
AUTH_TOKEN = os.environ.get("WRAPPER_AUTH_TOKEN", "")
 
 
class TaskRequest(BaseModel):
    task: str
 
 
@app.get("/health")
async def health():
    return {"status": "ok"}
 
 
@app.post("/run")
async def run_task(req: TaskRequest, x_auth_token: str = Header(default="")):
    if AUTH_TOKEN and x_auth_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="token inválido")
 
    if not req.task or not req.task.strip():
        raise HTTPException(status_code=400, detail="task vazia")
 
    agent = JarvisEngine()
    try:
        result = await agent.run(req.task)
        return {"result": result}
    except Exception as e:
        logger.error(f"erro executando tarefa: {e}")
        raise HTTPException(status_code=500, detail=str(e))
