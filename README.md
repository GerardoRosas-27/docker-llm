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

La primera vez verás la descarga en Descargas en curso (unos 1.4 GB). Al terminar pasa a **En marcha** sin pulsar nada. Si el modelo no está en memoria cuando escribes en el chat, el chat lo carga primero y enseña los segundos que lleva.

Ninguna respuesta se queda colgada: `GENERATION_TIMEOUT` (180 s) corta la respuesta en el servidor, `STREAM_IDLE_TIMEOUT` (90 s) la corta si no llega ningún token, y el navegador tiene su propio límite. En los tres casos el chat enseña el motivo. El botón Enviar pasa a **Detener** mientras genera.

Los archivos quedan en `./data` y no se suben a git.

## API de cada modelo

Con el modelo descargado, esta URL lo carga si hace falta y responde:

```bash
curl http://localhost:8080/v1/models/qwen-qwen3.5-2b-q4-k-m/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hola\"}]}"
```

También es compatible con el cliente de OpenAI. Desde otro proyecto:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
reply = client.chat.completions.create(
    model="qwen-qwen3.5-2b-q4-k-m",
    messages=[{"role": "user", "content": "Hola"}],
)
print(reply.choices[0].message.content)
```

`GET /v1/models` lista los modelos ya descargados. La documentación interactiva está en http://localhost:8080/docs.

Con `MASTER_SECRET` definido, las rutas `/v1` piden `Authorization: Bearer <API key>` (ver [Acceso](#acceso-master_secret)). Sin él, en local, van abiertas.

## Acceso (MASTER_SECRET)

Un solo secreto en las variables de entorno protege todo. De él salen, con HKDF-SHA256, las claves que firman las sesiones del panel y las API keys. Sin el secreto nadie puede entrar al panel ni crear claves. Si lo cambias, todas las sesiones y todas las API keys dejan de valer al momento.

Generar un secreto fuerte (48 bytes aleatorios, 64 caracteres):

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
# o bien
openssl rand -base64 48
```

Guárdalo en `MASTER_SECRET` (Railway → Variables, o `.env` en local) y en tu gestor de contraseñas. El servidor nunca lo registra ni lo devuelve.

**En el panel.** Pulsa **Acceso**, escribe el secreto maestro y pulsa **Entrar**. El navegador guarda solo una sesión firmada que caduca a las 12 h (`SESSION_TTL_HOURS`); el secreto no se guarda. Arriba a la derecha se ve el estado de la sesión, y **Cerrar sesión** la revoca en el servidor. En la misma ventana, **Generar API key** (con nombre y caducidad opcional) enseña la clave una sola vez con un botón **Copiar**. La lista permite revocarlas.

**Desde la API.**

```bash
# 1. Sesión de administración
TOKEN=$(curl -s https://TU-URL/api/auth/login -H "Content-Type: application/json" \
  -d '{"secret":"'"$MASTER_SECRET"'"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")

# 2. API key para otro proyecto (expires_in_days es opcional)
curl -s https://TU-URL/api/keys -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"label":"mi-app","expires_in_days":90}'

# 3. Usarla
curl https://TU-URL/v1/models -H "Authorization: Bearer obk1.…"
```

| Ruta | Quién | Qué hace |
| --- | --- | --- |
| `POST /api/auth/login` `{secret}` | cualquiera | Devuelve `{token, expires_at}`. 5 fallos por IP bloquean con espera creciente (30 s … 15 min). Sin `MASTER_SECRET` responde 409 `master_secret_not_configured`. |
| `POST /api/auth/logout` | sesión | Revoca la sesión. |
| `GET /api/auth/status` | cualquiera | Modo (`master`, `legacy`, `open`), avisos y la sesión enviada. |
| `GET/POST /api/keys`, `DELETE /api/keys/{id}` | sesión | Lista, crea y revoca API keys. La clave solo sale al crearla; la base guarda el id, no la clave. |
| `POST /api/access/generate` | sesión | Ruta antigua: ahora crea una API key derivada. |

La sesión se manda como `Authorization: Bearer` o `X-Admin-Token`. Una sesión de administración también vale para `/v1` (así funciona el chat del panel); una API key no vale para `/api`.

**Compatibilidad.** `ADMIN_TOKEN` y `API_KEY` siguen funcionando igual, también junto a `MASTER_SECRET`. Las claves que generaba la versión anterior del panel (guardadas en la base) solo valen mientras no haya `MASTER_SECRET`. Sin `MASTER_SECRET` ni claves, el panel queda abierto y lo avisa en `/health` y arriba del panel.

## Catálogo

La pestaña Catálogo busca modelos GGUF en Hugging Face. Cada archivo muestra su peso. Los que pasan de 9.00 GB salen deshabilitados, y el servidor rechaza esa descarga aunque alguien llame la API a mano. Solo se aceptan archivos `.gguf`.

## Railway

El mismo Dockerfile sirve para Railway:

1. Conecta este repositorio.
2. Monta un volumen en `/data` (Settings → Volumes). Sin volumen funciona igual, pero el modelo (1.4 GB) se vuelve a descargar en cada despliegue. El Dockerfile no lleva `VOLUME`: Railway rechaza esa instrucción y el build fallaba.
3. Con 2 GB de RAM basta para el modelo incluido. Los hilos de llama-server se ajustan a la cuota de CPU del servicio.
4. `PORT` lo pone Railway. La salud es `GET /health`.
5. Define `MASTER_SECRET` antes de exponer la URL (o, como antes, `ADMIN_TOKEN` y `API_KEY`).

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
| `MASTER_SECRET` | vacío | Secreto único para sesiones y API keys (recomendado) |
| `SESSION_TTL_HOURS` | 12 | Duración de la sesión del panel |
| `ADMIN_TOKEN` / `API_KEY` | vacío | Claves fijas antiguas; siguen funcionando |

## Pruebas

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```
