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

Si defines `API_KEY`, las rutas `/v1` piden `Authorization: Bearer …`. `ADMIN_TOKEN` protege el panel y `/api`. En local ambas van vacías.

## Catálogo

La pestaña Catálogo busca modelos GGUF en Hugging Face. Cada archivo muestra su peso. Los que pasan de 9.00 GB salen deshabilitados, y el servidor rechaza esa descarga aunque alguien llame la API a mano. Solo se aceptan archivos `.gguf`.

## Railway

El mismo Dockerfile sirve para Railway:

1. Conecta este repositorio.
2. Monta un volumen en `/data` (Settings → Volumes). Sin volumen funciona igual, pero el modelo (1.4 GB) se vuelve a descargar en cada despliegue. El Dockerfile no lleva `VOLUME`: Railway rechaza esa instrucción y el build fallaba.
3. Con 2 GB de RAM basta para el modelo incluido. Los hilos de llama-server se ajustan a la cuota de CPU del servicio.
4. `PORT` lo pone Railway. La salud es `GET /health`.
5. Define `API_KEY` y `ADMIN_TOKEN` antes de exponer la URL.

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

## Pruebas

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```
