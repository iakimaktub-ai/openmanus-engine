# OpenManus Engine for Jarvis

API HTTP simples que expõe o agente OpenManus (Manus) pro seu Jarvis pessoal chamar
como ferramenta, usando sua própria chave do Gemini como cérebro do agente.

Deploy: serviço Docker na Render.

## Configuração (dashboard da Render → seu serviço → aba "Environment")

- `GEMINI_API_KEY` (secret) — sua chave do Google AI Studio
- `WRAPPER_AUTH_TOKEN` (secret) — invente uma senha qualquer; use o mesmo valor no Jarvis
- `OPENMANUS_MODEL` (variável opcional) — padrão: gemini-3-flash-preview
- `VAULT_GITHUB_REPO` (variável opcional) — repositório `owner/repo` do vault de memória do Jarvis (ex.: `seu-usuario/jarvis-vault`); sem isso, a memória persistente fica desativada
- `VAULT_GITHUB_TOKEN` (secret, opcional) — GitHub Personal Access Token fine-grained, restrito a esse repositório, com permissão apenas de "Contents: Read and write"
- `VAULT_DIR` (variável opcional) — pasta local do checkout do vault dentro do container; padrão: `/tmp/jarvis_vault`

## Uso

POST /run
Headers: x-auth-token: SEU_TOKEN
Body: {"task": "descreva a tarefa aqui"}
