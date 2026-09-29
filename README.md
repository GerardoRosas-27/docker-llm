# Obrador

Panel en Docker para buscar modelos GGUF abiertos, descargar los que pesan **9 GB o menos**, ejecutarlos con llama.cpp y hablar con cada uno por su propia API.

La API está hecha en Python (FastAPI). La interfaz es el panel de administración: catálogo, descargas, arranque y un chat para probar cada modelo. Todo el cómputo corre dentro del contenedor. Los pesos se guardan en un volumen, no en la imagen.

## Modelo incluido

Al primer arranque se descarga solo, se comprueba su sha256 y se deja cargado en memoria:

| | |
| --- | --- |
| Repositorio | `bartowski/Qwen_Qwen3.5-2B-GGUF` |
| Archivo | `Qwen_Qwen3.5-2B-Q4_K_M.gguf` |
| Cuantización | Q4_K_M |
| Peso real | **1.40 GB** (1 396 198 496 bytes) |
| sha256 | `57a1085840f497d764a7fc5d346922dbde961efb54cc792ea81d694fd846a1d8` |
| RAM en uso | ~1.5 GB (llama-server) + ~70 MB (API) |
| Velocidad en CPU | ~10 tokens/s con 2 vCPU, ~28 tokens/s con 8 |
| Base | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), licencia Apache 2.0 |

URL de descarga: https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/resolve/main/Qwen_Qwen3.5-2B-Q4_K_M.gguf

Antes venía Qwen 3.5 9B Q4_K_M (6.17 GB). Pedía unos 7-8 GB de RAM y en CPU va unas 4-5 veces más lento que el 2B (pocos tokens por segundo), así que en Docker Desktop o en un plan pequeño el chat se quedaba esperando sin decir nada. Los volúmenes que ya tenían el 9B reciben el 2B al arrancar la nueva imagen; el 9B sigue en Instalados y se puede borrar desde ahí. Si el contenedor no tiene RAM para un modelo, el panel lo avisa y no lo arranca (`RAM_CHECK=0` para forzarlo).

## Arranque local

Docker Desktop tiene que estar en marcha.

```bash
docker compose up --build
```

Panel: http://localhost:8080

Antes define `MASTER_SECRET` en un `.env` junto al `docker-compose.yml` (ver [Acceso](#acceso-solo-master_secret)); sin él el panel y la API quedan cerrados.

La primera vez verás la descarga en Descargas en curso (unos 1.4 GB). Al terminar pasa a **En marcha** sin pulsar nada. Si el modelo no está en memoria cuando escribes en el chat, el chat lo carga primero y enseña los segundos que lleva.

Ninguna respuesta se queda colgada: `GENERATION_TIMEOUT` (180 s) corta la respuesta en el servidor, `STREAM_IDLE_TIMEOUT` (90 s) la corta si no llega ningún token, y el navegador tiene su propio límite. En los tres casos el chat enseña el motivo. El botón Enviar pasa a **Detener** mientras genera.

Los archivos quedan en `./data` y no se suben a git.

## Acceso (solo MASTER_SECRET)

Hay una sola forma de entrar: el secreto maestro `MASTER_SECRET`. De él salen, con HKDF-SHA256, las claves que firman las sesiones del panel y las API keys. Sin el secreto nadie puede entrar al panel ni crear claves. Si lo cambias, todas las sesiones y todas las API keys dejan de valer al momento.

**Sin `MASTER_SECRET` todo queda cerrado.** El panel carga pero no deja hacer nada, `/api/*` y `/v1/*` responden `503` con `"code": "master_secret_not_configured"` y `GET /health` (siempre público) lo avisa en `auth.warnings`. No existe modo abierto. `ADMIN_TOKEN`, `API_KEY` y las claves que generaba la versión antigua del panel ya no dan acceso: se pueden borrar de las variables.

Generar un secreto fuerte (48 bytes aleatorios, 64 caracteres):

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
# o bien
openssl rand -base64 48
```

Guárdalo en `MASTER_SECRET` (Railway → Variables, o `.env` en local) y en tu gestor de contraseñas. El servidor nunca lo registra ni lo devuelve.

**En el panel.** Pulsa **Acceso**, escribe el secreto maestro y pulsa **Entrar**. El navegador guarda solo una sesión firmada que caduca a las 12 h (`SESSION_TTL_HOURS`); el secreto no se guarda. Arriba a la derecha se ve el estado de la sesión, y **Cerrar sesión** la revoca en el servidor. En la misma ventana, **Generar API key** (con nombre y caducidad opcional) enseña la clave una sola vez, con un botón **Copiar** y los curl listos para cada modelo descargado. La lista permite revocarlas. En **Instalados**, cada modelo tiene su sección **Usar por API** con el curl (normal y streaming), el listado de modelos y el ejemplo del cliente de OpenAI, ya con la URL real del servicio.

**Desde la API.**

```bash
BASE=https://docker-llm-production.up.railway.app   # o http://localhost:8080

# 1. Sesión de administración
TOKEN=$(curl -s $BASE/api/auth/login -H "Content-Type: application/json" \
  -d '{"secret":"'"$MASTER_SECRET"'"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")

# 2. API key para otro proyecto (expires_in_days es opcional)
curl -s $BASE/api/keys -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"label":"mi-app","expires_in_days":90}'
# -> {"id": "...", "label": "mi-app", ..., "key": "obk1.…"}  (la clave solo sale aquí)
```

| Ruta | Quién | Qué hace |
| --- | --- | --- |
| `GET /health` | cualquiera | Estado. `auth.mode` es `master` o `unconfigured`, con avisos. |
| `POST /api/auth/login` `{secret}` | cualquiera | Devuelve `{token, expires_at}`. 5 fallos por IP bloquean con espera creciente (30 s … 15 min). |
| `POST /api/auth/logout` | sesión | Revoca la sesión. |
| `GET /api/auth/status` | cualquiera | Modo, avisos y la sesión enviada. |
| `GET/POST /api/keys`, `DELETE /api/keys/{id}` | sesión | Lista, crea y revoca API keys. La base guarda el id, no la clave. |
| resto de `/api/*` | sesión | Administración del panel. |
| `/v1/*` | API key (o la sesión del panel, para su chat) | Consumo de los modelos. |

Todo va en `Authorization: Bearer …`. Una API key (`obk1.…`) vale para todas las rutas `/v1` y nunca para `/api`.

## Consumir los modelos por API

Las API keys generadas son las claves de los modelos desplegados. Valen para:

- `GET /v1/models` y `GET /v1/models/{id}`
- `POST /v1/chat/completions` (con `"stream": true` devuelve SSE)
- `POST /v1/completions`
- `POST /v1/models/{id}/chat/completions` y `POST /v1/models/{id}/completions` (ruta propia de cada modelo; el `model` del cuerpo no hace falta)

El id de cada modelo sale en `GET /v1/models` y en el panel (Instalados). Si el modelo no está en memoria, la primera llamada lo carga. Ejemplos con el modelo incluido, `qwen-qwen3.5-2b-q4-k-m` (para otro modelo cambia solo el id):

```bash
export BASE=https://docker-llm-production.up.railway.app   # o http://localhost:8080
export API_KEY="obk1.…"                                     # la que te dio Generar API key

# Modelos disponibles
curl $BASE/v1/models \
  -H "Authorization: Bearer $API_KEY"

# Chat (respuesta completa)
curl $BASE/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-qwen3.5-2b-q4-k-m","messages":[{"role":"user","content":"Hola"}]}'

# Chat en streaming (SSE, token a token)
curl -N $BASE/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-qwen3.5-2b-q4-k-m","stream":true,"messages":[{"role":"user","content":"Hola"}]}'

# Ruta propia del modelo (equivalente)
curl $BASE/v1/models/qwen-qwen3.5-2b-q4-k-m/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hola"}]}'

# Completions (texto libre)
curl $BASE/v1/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-qwen3.5-2b-q4-k-m","prompt":"Érase una vez","max_tokens":40}'
```

Sin clave, o con una clave revocada, caducada o de otro secreto, la respuesta es `401` con `{"error": {"code": "invalid_api_key", ...}}`.

Con el cliente de OpenAI, desde otro proyecto:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://docker-llm-production.up.railway.app/v1",
    api_key="obk1.…",  # API key generada en el panel
)
reply = client.chat.completions.create(
    model="qwen-qwen3.5-2b-q4-k-m",
    messages=[{"role": "user", "content": "Hola"}],
)
print(reply.choices[0].message.content)

# En streaming
for chunk in client.chat.completions.create(
    model="qwen-qwen3.5-2b-q4-k-m",
    messages=[{"role": "user", "content": "Hola"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

La documentación interactiva está en `/docs`.

## Catálogo

La pestaña Catálogo busca modelos GGUF en Hugging Face. Cada archivo muestra su peso. Los que pasan de 9.00 GB salen deshabilitados, y el servidor rechaza esa descarga aunque alguien llame la API a mano. Solo se aceptan archivos `.gguf`.

## Railway

El mismo Dockerfile sirve para Railway:

1. Conecta este repositorio.
2. Monta un volumen en `/data` (Settings → Volumes). Sin volumen funciona igual, pero el modelo (1.4 GB) se vuelve a descargar en cada despliegue. El Dockerfile no lleva `VOLUME`: Railway rechaza esa instrucción y el build fallaba.
3. Con 2 GB de RAM basta para el modelo incluido. Los hilos de llama-server se ajustan a la cuota de CPU del servicio.
4. `PORT` lo pone Railway. La salud es `GET /health`.
5. Define `MASTER_SECRET` en Variables. Sin él el servicio arranca, pero el panel y `/v1` responden 503. `ADMIN_TOKEN` y `API_KEY` ya no se usan: bórralos.

`railway.toml` ya apunta al Dockerfile y al healthcheck.

## Variables

Están comentadas en `.env.example`. Las que más importan:

| Variable | Defecto | Para qué |
| --- | --- | --- |
| `DEFAULT_REPO` / `DEFAULT_FILE` | Qwen 3.5 2B Q4_K_M | Modelo que se descarga solo |
| `PRELOAD_DEFAULT` | 1 | Deja cargado el modelo por defecto |
| `CTX_SIZE` | 2048 | Contexto |
| `GENERATION_TIMEOUT` | 180 | Segundos máximos por respuesta |
| `STREAM_IDLE_TIMEOUT` | 90 | Segundos máximos sin tokens |
| `LOAD_TIMEOUT` | 600 | Segundos máximos para cargar un modelo |
| `LLAMA_THREADS` | 0 (auto) | Hilos de llama-server |
| `LLAMA_REPACK` | 0 | 1 duplica la RAM de los pesos por algo más de velocidad |
| `CACHE_RAM_MIB` | 0 | Caché de prompts de llama-server (su defecto son 8 GiB) |
| `RAM_CHECK` | 1 | No arranca modelos que no caben en la RAM |
| `MALLOC_ARENA_MAX` | 2 | Menos RSS en glibc |
| `MASTER_SECRET` | vacío | **Obligatorio.** Secreto único para sesiones y API keys. Vacío = todo cerrado (503) |
| `SESSION_TTL_HOURS` | 12 | Duración de la sesión del panel |

## Pruebas

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```
