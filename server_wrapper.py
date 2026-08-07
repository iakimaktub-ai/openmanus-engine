import os
import io
import wave
import base64
import asyncio
import subprocess
import time
import uuid
from typing import Optional
import httpx
import psutil

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
        'country = "br"\n\n'
        "[browser]\n"
        "headless = true\n"
        "disable_security = true\n"
    )
    with open("config/config.toml", "w") as f:
        f.write(toml_content)

_write_config()

import re


def _extract_final_answer(raw: str) -> str:
    """O agent.run() do OpenManus devolve o histórico INTEIRO de passos concatenado
    (assim a biblioteca funciona por padrão) — não só a resposta final. E o tool
    `terminate` não carrega um resumo, só confirma "concluído com sucesso". Então
    a resposta de verdade fica no último passo útil ANTES do terminate. Esta função
    filtra o log bruto pra extrair só essa parte, com fallback pro texto cru caso
    o formato não bata com o esperado (evita quebrar caso a lib mude o log)."""
    if not raw:
        return raw
    try:
        parts = re.split(r"Step \d+: ", raw)
        parts = [p.strip() for p in parts if p.strip()]
        # descarta o passo do terminate (genérico, sem conteúdo útil) e passos de erro
        useful = [
            p for p in parts
            if "cmd `terminate`" not in p and not p.startswith("Error:")
        ]
        if not useful:
            return raw.strip()
        last = useful[-1]
        # remove o prefixo padrão "Observed output of cmd `x` executed:"
        last = re.sub(r"^Observed output of cmd `[^`]+` executed:\s*", "", last)
        # se veio como dict Python (ex.: {'observation': '...', 'success': True}),
        # extrai só o texto de 'observation'
        m = re.search(r"'observation':\s*'(.*)',\s*'success'", last, re.DOTALL)
        if m:
            obs = m.group(1)
            if "\\n" in obs:
                try:
                    obs = obs.encode().decode("unicode_escape")
                except Exception:
                    pass
            return obs.strip()
        return last.strip()
    except Exception:
        return raw.strip()


# --- agora sim, o resto dos imports ---
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.agent.toolcall import ToolCallAgent
from app.tool import (
    Bash,
    BrowserUseTool,
    PlanningTool,
    StrReplaceEditor,
    Terminate,
    ToolCollection,
    WebSearch,
)
from app.tool.python_execute import PythonExecute
from app.logger import logger
from browser_use import Browser as BrowserUseBrowser, BrowserConfig


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
        "REGRA DE FORMATO: o texto que você produzir na SUA ÚLTIMA AÇÃO informativa antes de chamar "
        "terminate (ex.: o texto que um script Python imprime, ou a mensagem final que você escrever) "
        "é o que será mostrado E FALADO em voz alta para a pessoa — então essa última informação deve "
        "estar em formato de fala natural: frases corridas, sem markdown, sem `====`, sem emojis, sem "
        "listas com marcadores. Só números e valores relevantes, ditos como numa conversa. "
        "Ao concluir, chame a ferramenta terminate — ela apenas encerra a execução, então garanta que a "
        "resposta final já foi dada de forma completa e falável na ação anterior. "
        "Quando o usuário pedir para ver, abrir ou mostrar um site ou página específica, inclua no final "
        "da sua resposta a tag [[OPEN_PANEL:url|título]], onde 'url' é o endereço completo (com https://) "
        "e 'título' é um nome curto para o painel. Não use essa tag se o usuário não pediu para ver algo "
        "visualmente. Nunca explique a tag ao usuário, ela é removida antes de ser exibida. "
        "Você agora também pode navegar de verdade em sites usando a ferramenta browser_use (abrir "
        "páginas, clicar, preencher campos, seguir links). REGRA DE SEGURANÇA (obrigatória, sem "
        "exceção): antes de executar qualquer ação que mude o estado do site — enviar um formulário, "
        "fazer login, finalizar uma compra, publicar/postar algo, excluir algo — PARE e chame "
        "terminate descrevendo exatamente qual ação executaria e por quê, SEM executá-la. Só execute "
        "essa ação sensível se a tarefa recebida disser explicitamente que o usuário já confirmou."
    )
    next_step_prompt: str = (
        "Escolha a ferramenta mais adequada para avançar a tarefa. "
        "Quando a tarefa estiver completa, chame terminate com o resumo do resultado."
    )

    max_steps: int = 20

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            Bash(),
            StrReplaceEditor(),
            WebSearch(),
            BrowserUseTool(),
            PlanningTool(),
            Terminate(),
        )
    )
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    # True só para o agente da sessão de navegador ao vivo (ver _start_browser_session).
    # ToolCallAgent.run() chama self.cleanup() em um `finally` ao fim de TODA tarefa,
    # e o cleanup() padrão da BrowserUseTool fecha o Chromium/contexto de verdade
    # (não só solta a referência) — isso apagava o browser compartilhado a cada
    # chamada de /run, mesmo com o painel ainda aberto (a tela VNC ficava preta
    # de novo pouco depois de renderizar a página real). Aqui pulamos só a
    # ferramenta browser_use nesse caso; o browser é fechado explicitamente em
    # _stop_browser_session() quando a sessão de fato termina.
    keep_browser_alive: bool = False

    async def cleanup(self):
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if self.keep_browser_alive and tool_name == "browser_use":
                continue
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(f"erro limpando ferramenta '{tool_name}': {e}", exc_info=True)


# --- sessão de navegador ao vivo (Xvfb + Chrome headful + x11vnc) ---
# Uma única sessão global ativa por vez (fora de escopo do design: múltiplas
# sessões simultâneas). display/porta fixos porque só existe uma sessão.
BROWSER_DISPLAY = ":99"
BROWSER_VNC_PORT = 5999
# 180s (3 min) em vez de 300s: no plano free do Render (limite de 512MB), cada
# minuto extra com Xvfb+x11vnc+Chromium headful ativos e ociosos é memória que
# poderia já ter sido liberada — reduz a chance de OOM sem prejudicar o uso real.
BROWSER_SESSION_TIMEOUT_SECONDS = 180

_browser_session = {
    "session_id": None,
    "agent": None,
    "xvfb_proc": None,
    "x11vnc_proc": None,
    "last_activity": 0.0,
}
_browser_session_lock = asyncio.Lock()


async def _start_browser_session() -> str:
    """Sobe Xvfb + x11vnc e cria um JarvisEngine com o Chrome em modo headful
    apontando pro display virtual. Idempotente: se já existe sessão ativa,
    devolve o session_id existente em vez de subir tudo de novo."""
    if _browser_session["session_id"]:
        return _browser_session["session_id"]

    # resolução/profundidade de cor reduzidas (era 1280x800x24): o noVNC do painel
    # escala automaticamente pro tamanho reportado pelo servidor VNC, então isso
    # não quebra a exibição — só reduz o framebuffer do Xvfb e o tráfego do x11vnc.
    xvfb_proc = subprocess.Popen(
        ["Xvfb", BROWSER_DISPLAY, "-screen", "0", "1024x768x16"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # dá tempo do Xvfb terminar de subir antes do x11vnc/Chrome tentarem usar o display
    await asyncio.sleep(1.5)

    x11vnc_proc = subprocess.Popen(
        [
            "x11vnc", "-display", BROWSER_DISPLAY,
            "-rfbport", str(BROWSER_VNC_PORT),
            "-forever", "-shared", "-nopw", "-quiet",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    os.environ["DISPLAY"] = BROWSER_DISPLAY
    agent = JarvisEngine()
    agent.keep_browser_alive = True
    live_tool = agent.available_tools.get_tool("browser_use")
    live_tool.browser = BrowserUseBrowser(BrowserConfig(
        headless=False,
        disable_security=True,
        # flags de baixo consumo de memória: no plano free do Render (limite de
        # 512MB) o Chromium headful sozinho já passava de 300-400MB, o que somado
        # ao Python/Xvfb/x11vnc estourava o limite (evento real: "Ran out of
        # memory (used over 512MB)"). Essas flags desligam GPU/compositor,
        # rede/telemetria em segundo plano e limitam o heap do V8, sem afetar a
        # navegação em si.
        extra_chromium_args=[
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-software-rasterizer",
            "--disable-extensions",
            "--disable-background-networking",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-breakpad",
            "--disable-component-update",
            "--disable-default-apps",
            "--disable-sync",
            "--disable-translate",
            "--metrics-recording-only",
            "--mute-audio",
            "--no-first-run",
            "--renderer-process-limit=1",
            "--js-flags=--max-old-space-size=256",
        ],
    ))

    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    _browser_session.update({
        "session_id": session_id,
        "agent": agent,
        "xvfb_proc": xvfb_proc,
        "x11vnc_proc": x11vnc_proc,
        "last_activity": time.time(),
    })
    return session_id


async def _stop_browser_session():
    """Fecha o browser_use de verdade (Chromium/contexto), mata Xvfb/x11vnc e
    limpa o estado global da sessão. Precisa fechar o browser aqui porque, com
    keep_browser_alive=True, o cleanup automático do agente pula essa tool."""
    agent = _browser_session.get("agent")
    if agent is not None:
        browser_tool = agent.available_tools.get_tool("browser_use")
        if browser_tool is not None:
            try:
                await browser_tool.cleanup()
            except Exception as e:
                logger.error(f"erro fechando browser da sessão: {e}", exc_info=True)
    for key in ("x11vnc_proc", "xvfb_proc"):
        proc = _browser_session.get(key)
        if proc is not None and proc.poll() is None:
            proc.terminate()
    _browser_session.update({
        "session_id": None,
        "agent": None,
        "xvfb_proc": None,
        "x11vnc_proc": None,
        "last_activity": 0.0,
    })


async def _browser_session_reaper():
    """Encerra a sessão de navegador se ficar 5 min sem atividade (chamada de
    browse_website nem tráfego WebSocket do painel ao vivo)."""
    while True:
        await asyncio.sleep(30)
        session_id = _browser_session["session_id"]
        if session_id and (time.time() - _browser_session["last_activity"]) > BROWSER_SESSION_TIMEOUT_SECONDS:
            logger.info(f"encerrando sessão de navegador {session_id} por inatividade")
            await _stop_browser_session()


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


@app.on_event("startup")
async def _on_startup():
    asyncio.create_task(_browser_session_reaper())

# Token simples pra ninguém além do seu Jarvis conseguir usar sua chave/quota.
# Configure o mesmo valor como Secret WRAPPER_AUTH_TOKEN aqui no Space, e
# cole esse mesmo valor no campo de token do Jarvis.
AUTH_TOKEN = os.environ.get("WRAPPER_AUTH_TOKEN", "").strip()


class TaskRequest(BaseModel):
    task: str
    session_id: Optional[str] = None


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


@app.get("/stats")
async def stats(x_auth_token: str = Header(default="")):
    if AUTH_TOKEN and x_auth_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="token inválido")
    return {
        "cpu_percent": psutil.cpu_percent(interval=0),
        "memory_percent": psutil.virtual_memory().percent,
    }


@app.post("/browse/start")
async def browse_start(x_auth_token: str = Header(default="")):
    if AUTH_TOKEN and x_auth_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="token inválido")
    async with _browser_session_lock:
        try:
            session_id = await _start_browser_session()
        except Exception as e:
            logger.error(f"erro iniciando sessão de navegador ao vivo: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    return {"session_id": session_id}


@app.post("/browse/stop")
async def browse_stop(x_auth_token: str = Header(default="")):
    if AUTH_TOKEN and x_auth_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="token inválido")
    async with _browser_session_lock:
        await _stop_browser_session()
    return {"status": "ok"}


@app.websocket("/browse/ws/{session_id}")
async def browse_ws(websocket: WebSocket, session_id: str):
    token = websocket.query_params.get("token", "")
    if AUTH_TOKEN and token != AUTH_TOKEN:
        await websocket.close(code=4401)
        return
    if _browser_session["session_id"] != session_id:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", BROWSER_VNC_PORT)
    except OSError as e:
        logger.error(f"erro conectando ao x11vnc local: {e}")
        await websocket.close(code=1011)
        return

    async def ws_to_vnc():
        try:
            while True:
                data = await websocket.receive_bytes()
                _browser_session["last_activity"] = time.time()
                writer.write(data)
                await writer.drain()
        except WebSocketDisconnect:
            pass

    async def vnc_to_ws():
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                _browser_session["last_activity"] = time.time()
                await websocket.send_bytes(data)
        except Exception:
            pass

    task_a = asyncio.create_task(ws_to_vnc())
    task_b = asyncio.create_task(vnc_to_ws())
    done, pending = await asyncio.wait({task_a, task_b}, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    writer.close()


@app.post("/run")
async def run_task(req: TaskRequest, x_auth_token: str = Header(default="")):
    if AUTH_TOKEN and x_auth_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="token inválido")

    if not req.task or not req.task.strip():
        raise HTTPException(status_code=400, detail="task vazia")

    if req.session_id:
        if _browser_session["session_id"] != req.session_id:
            raise HTTPException(
                status_code=410,
                detail="sessão de navegador expirada ou inexistente; abra o painel de navegador novamente",
            )
        agent = _browser_session["agent"]
        _browser_session["last_activity"] = time.time()
    else:
        agent = JarvisEngine()

    try:
        raw_result = await agent.run(req.task)
        result = _extract_final_answer(raw_result)
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
