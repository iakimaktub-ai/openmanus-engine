"""
OpenManus Engine for Jarvis — server_wrapper.py
API HTTP simples que expõe:
  - POST /run    -> agente que usa o Gemini como "cérebro" para executar/raciocinar sobre uma tarefa
  - POST /speak  -> converte texto em áudio usando a voz clonada "Voz do Jarvis" no Higgsfield

Variáveis de ambiente (Settings -> Variables and secrets do Space):
  GEMINI_API_KEY      (secret, obrigatório)   -> sua chave do Google AI Studio
  WRAPPER_AUTH_TOKEN  (secret, obrigatório)   -> senha inventada por você; use o mesmo valor no Jarvis
  OPENMANUS_MODEL     (variável, opcional)    -> padrão: gemini-3-flash-preview
  HF_CREDENTIALS      (secret, obrigatório para /speak) -> "KEY_ID:KEY_SECRET" da Higgsfield Cloud
"""

import os
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
WRAPPER_AUTH_TOKEN = os.environ.get("WRAPPER_AUTH_TOKEN", "")
OPENMANUS_MODEL = os.environ.get("OPENMANUS_MODEL", "gemini-3-flash-preview")
HF_CREDENTIALS = os.environ.get("HF_CREDENTIALS", "")

HIGGSFIELD_VOICE_ID = "bd7393a2-5a47-4f91-b516-d888dc92670c"  # "Voz do Jarvis"

app = FastAPI(title="OpenManus Engine for Jarvis")

# Libera chamadas vindas do arquivo jarvis.html (rodando local no navegador, sem origem "https://")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_auth(request: Request):
    token = request.headers.get("x-auth-token")
    if not WRAPPER_AUTH_TOKEN or token != WRAPPER_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="x-auth-token inválido ou ausente")


@app.get("/")
async def health():
    return {"status": "ok", "service": "openmanus-engine-for-jarvis"}


@app.post("/run")
async def run(request: Request):
    check_auth(request)
    body = await request.json()
    task = (body or {}).get("task", "").strip()
    if not task:
        raise HTTPException(status_code=400, detail="campo 'task' é obrigatório")
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY não configurada no Space")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{OPENMANUS_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "Você é um agente de execução de tarefas (estilo OpenManus/Manus). "
                            "Receba a tarefa abaixo, raciocine passo a passo se precisar, e devolva "
                            "uma resposta final clara e direta, em português do Brasil.\n\n"
                            f"TAREFA: {task}"
                        )
                    }
                ]
            }
        ]
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, json=payload)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Erro ao chamar o Gemini (HTTP {resp.status_code}): {resp.text[:300]}",
        )

    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        text = "Não consegui extrair uma resposta do modelo."

    return {"result": text}


@app.post("/speak")
async def speak(request: Request):
    check_auth(request)
    body = await request.json()
    text = (body or {}).get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="campo 'text' é obrigatório")
    if not HF_CREDENTIALS:
        raise HTTPException(status_code=500, detail="HF_CREDENTIALS não configurada no Space")

    os.environ["HF_CREDENTIALS"] = HF_CREDENTIALS  # garante que o SDK enxergue a credencial
    try:
        import higgsfield_client
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="pacote 'higgsfield-client' não instalado — confira o requirements.txt",
        )

    try:
        result = await higgsfield_client.subscribe_async(
            "seed_audio",
            arguments={
                "prompt": text,
                "voice_type": "element",
                "voice_id": HIGGSFIELD_VOICE_ID,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erro na Higgsfield: {e}")

    audio_url = None
    if isinstance(result, dict):
        for key in ("audio", "audios", "output", "outputs"):
            val = result.get(key)
            if isinstance(val, list) and val:
                audio_url = val[0].get("url") or val[0].get("audio_url")
                break
        if not audio_url:
            audio_url = result.get("url") or result.get("audio_url")

    if not audio_url:
        raise HTTPException(status_code=502, detail=f"Resposta inesperada da Higgsfield: {result}")

    return {"audio_url": audio_url}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
