import os
import io
import wave
import base64
import httpx

# --- gera config/config.toml a partir de variáveis de ambiente (Secrets do Space) ---
# precisa acontecer ANTES de importar qualquer coisa de app.* (o OpenManus lê o
# config.toml no momento em que o pacote app.config é importado).

def _write_config():
    # .strip() remove espaços em branco e quebras de linha acidentais que às vezes
    # vêm junto ao copiar/colar a chave de um lugar pra outro (evita quebrar o TOML).
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    model = os.environ.get("OPENMANUS_MODEL", "claude-sonnet-5").strip()
    base_url = "https://api.anthropic.com/v1/"
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
        'daytona_api_key = "unused"\n\n'
        "[search]\n"
        'engine = "Bing"\n'
        "fallback_engines = [\"DuckDuckGo\"]\n"
        'lang = "pt"\n'
        'country = "br"\n'
    )
    with open("config/config.toml", "w") as f:
        f.write(toml_content)

_write_config()

# --- agora sim, o resto dos imports ---
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
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
        "Seja direto e eficiente. "
        "REGRA DE IDIOMA (obrigatória, sem exceção): mesmo que os resultados de busca, artigos ou "
        "qualquer fonte consultada estejam em inglês ou outro idioma, TRADUZA tudo e escreva o raciocínio "
        "e principalmente o resumo final inteiramente em português do Brasil. Nunca cole trechos em inglês "
        "no resumo final, nem misture os dois idiomas na mesma frase. "
        "Ao concluir, chame a ferramenta terminate com um resumo claro, 100% em português, do "
        "resultado final — esse resumo é o que será mostrado (e falado em voz) à pessoa."
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
# Origens liberadas. TEMPORARIAMENTE em "*" (qualquer origem) para confirmar que o
# CORS é mesmo a causa. Depois de testar com sucesso, trocar para a lista específica:
# ["null", "https://iakimaktub-ai.github.io"]
ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Token simples pra ninguém além do seu Jarvis conseguir usar sua chave/quota.
# Configure o mesmo valor como Secret WRAPPER_AUTH_TOKEN aqui no Space, e
# cole esse mesmo valor no campo de token do Jarvis.
AUTH_TOKEN = os.environ.get("WRAPPER_AUTH_TOKEN", "").strip()


class TaskRequest(BaseModel):
    task: str


class TTSRequest(BaseModel):
    text: str


GEMINI_TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
GEMINI_API_KEY_FOR_TTS = os.environ.get("GEMINI_API_KEY", "").strip()

HF_CREDENTIALS = os.environ.get("HF_CREDENTIALS", "").strip()
HIGGSFIELD_VOICE_ID = "bd7393a2-5a47-4f91-b516-d888dc92670c"  # "Voz do Jarvis"


def _pcm_to_wav_bytes(pcm_bytes, sample_rate=24000, channels=1, sample_width=2):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


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


@app.post("/tts")
async def text_to_speech(req: TTSRequest, x_auth_token: str = Header(default="")):
    if AUTH_TOKEN and x_auth_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="token inválido")
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="texto vazio")
    if not GEMINI_API_KEY_FOR_TTS:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY não configurada no servidor")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TTS_MODEL}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": "Fale de forma natural, calma e clara, em português do Brasil: " + req.text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Charon"}}},
        },
    }
    headers = {"content-type": "application/json", "x-goog-api-key": GEMINI_API_KEY_FOR_TTS}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            logger.error(f"erro TTS Gemini: {resp.status_code} {resp.text[:300]}")
            raise HTTPException(status_code=502, detail=f"Gemini TTS retornou {resp.status_code}")
        data = resp.json()
        part = data["candidates"][0]["content"]["parts"][0]
        inline = part.get("inlineData") or part.get("inline_data")
        if not inline or not inline.get("data"):
            raise HTTPException(status_code=502, detail="Gemini não retornou áudio")
        mime_type = inline.get("mimeType") or inline.get("mime_type") or "audio/L16;codec=pcm;rate=24000"
        sample_rate = 24000
        if "rate=" in mime_type:
            try:
                sample_rate = int(mime_type.split("rate=")[1].split(";")[0])
            except Exception:
                pass
        pcm_bytes = base64.b64decode(inline["data"])
        wav_bytes = _pcm_to_wav_bytes(pcm_bytes, sample_rate=sample_rate)
        return Response(content=wav_bytes, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"erro gerando TTS: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speak")
async def speak(req: TTSRequest, x_auth_token: str = Header(default="")):
    """Gera fala usando a voz clonada 'Voz do Jarvis' no Higgsfield."""
    if AUTH_TOKEN and x_auth_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="token inválido")
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="texto vazio")
    if not HF_CREDENTIALS:
        raise HTTPException(status_code=500, detail="HF_CREDENTIALS não configurada no servidor")

    # O SDK da Higgsfield não lê "HF_CREDENTIALS" diretamente — ele espera HF_KEY
    # (um valor só) ou HF_API_KEY + HF_API_SECRET (dois valores separados).
    # Aqui guardamos a credencial como "KEY_ID:KEY_SECRET" numa única secret e
    # separamos nas duas variáveis que o SDK realmente procura.
    if ":" in HF_CREDENTIALS:
        hf_key_id, hf_key_secret = HF_CREDENTIALS.split(":", 1)
        os.environ["HF_API_KEY"] = hf_key_id.strip()
        os.environ["HF_API_SECRET"] = hf_key_secret.strip()
    else:
        os.environ["HF_KEY"] = HF_CREDENTIALS.strip()
    try:
        import higgsfield_client
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="pacote 'higgsfield-client' não instalado — confira o requirements.txt",
        )

    try:
        import asyncio
        result = await asyncio.wait_for(
            higgsfield_client.subscribe_async(
                "text2speech_v2",
                arguments={
                    "prompt": req.text,
                    "variant": "elevenlabs",
                    "voice_type": "element",
                    "voice_id": HIGGSFIELD_VOICE_ID,
                },
            ),
            timeout=5.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Higgsfield demorou demais para responder (timeout de 8s)")
    except Exception as e:
        logger.error(f"erro na Higgsfield: {e} | detalhe completo: {repr(e)}")
        raise HTTPException(status_code=502, detail=f"Erro na Higgsfield: {e}")

    audio_url = None
    if isinstance(result, dict):
        audio_field = result.get("audio")
        if isinstance(audio_field, dict):
            audio_url = audio_field.get("url")
        elif isinstance(audio_field, list) and audio_field:
            audio_url = audio_field[0].get("url") or audio_field[0].get("audio_url")
        if not audio_url:
            for key in ("audios", "output", "outputs"):
                val = result.get(key)
                if isinstance(val, list) and val:
                    audio_url = val[0].get("url") or val[0].get("audio_url")
                    break
        if not audio_url:
            audio_url = result.get("url") or result.get("audio_url")

    if not audio_url:
        raise HTTPException(status_code=502, detail=f"Resposta inesperada da Higgsfield: {result}")

    return {"audio_url": audio_url}
