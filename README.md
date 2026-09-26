# Obrador

Panel en Docker para buscar modelos GGUF abiertos, descargar los que pesan **8 GB o menos**, ejecutarlos con llama.cpp y hablar con cada uno por su propia API.

La API está hecha en Python (FastAPI). La interfaz es el panel de administración: catálogo, descargas, arranque y un chat para probar cada modelo. Todo el cómputo corre dentro del contenedor. Los pesos se guardan en un volumen, no en la imagen.

## Modelo incluido

Al primer arranque se encola:

| | |
| --- | --- |
| Repositorio | `bartowski/Qwen_Qwen3.5-9B-GGUF` |
| Archivo | `Qwen_Qwen3.5-9B-Q4_K_M.gguf` |
| Cuantización | Q4_K_M |
| Peso real | **6.17 GB** (6 169 341 984 bytes) |
| Base | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), licencia Apache 2.0 |

Es el Q4 recomendado de Qwen 3.5 9B. En el mismo repositorio, `Q4_K_L` pesa 6.92 GB y también pasa el límite; `Q8_0` pesa 9.80 GB y el panel lo rechaza. El tamaño se vuelve a comprobar contra Hugging Face antes de cada descarga.

Hace falta espacio de sobra en el volumen (unos 8 GB libres) y RAM suficiente para cargarlo. En la práctica, el 9B en Q4 con contexto 2048 pide cerca de 8 GB de RAM dentro de Docker. Una máquina de 16 GB puede con un solo modelo. El panel mantiene un modelo en memoria a la vez.

## Arranque local

Docker Desktop tiene que estar en marcha.

```bash
docker compose up --build
```

Panel: http://localhost:8080

La primera vez verás la descarga de Qwen en la pestaña Instalados. Cuando pase a **Descargado**, pulsa **Arrancar** o abre el chat. En Docker Desktop para Windows esa primera carga leyó el archivo en unos 14 minutos y después respondió. El límite de espera es `LOAD_TIMEOUT` (20 minutos).

Los archivos quedan en `./data` y no se suben a git.

## API de cada modelo

Con el modelo descargado, esta URL lo carga si hace falta y responde:

```bash
curl http://localhost:8080/v1/models/qwen-qwen3.5-9b-q4-k-m/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hola\"}]}"
```

También es compatible con el cliente de OpenAI. Desde otro proyecto:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
reply = client.chat.completions.create(
    model="qwen-qwen3.5-9b-q4-k-m",
    messages=[{"role": "user", "content": "Hola"}],
)
print(reply.choices[0].message.content)
```

`GET /v1/models` lista los modelos ya descargados. La documentación interactiva está en http://localhost:8080/docs.

Si defines `API_KEY`, las rutas `/v1` piden `Authorization: Bearer …`. `ADMIN_TOKEN` protege el panel y `/api`. En local ambas van vacías.

## Catálogo

La pestaña Catálogo busca modelos GGUF en Hugging Face. Cada archivo muestra su peso. Los que pasan de 8.00 GB salen deshabilitados, y el servidor rechaza esa descarga aunque alguien llame la API a mano. Solo se aceptan archivos `.gguf`.

## Railway

El mismo Dockerfile sirve para Railway:

1. Conecta este repositorio.
2. Monta un volumen en `/data` (ahí viven los 6.17 GB y la base de datos).
3. El servicio tiene que ofrecer RAM de sobra para el 9B. Un plan pequeño no lo carga aunque la descarga quepa.
4. `PORT` lo pone Railway. La salud es `GET /health`.
5. Define `API_KEY` y `ADMIN_TOKEN` antes de exponer la URL.

`railway.toml` ya apunta al Dockerfile y al healthcheck.

## Variables

Están comentadas en `.env.example`. Las que más importan: `CTX_SIZE` (por defecto 2048), `N_GPU_LAYERS` (0, CPU) y `MAX_LOADED_MODELS` (1).

## Pruebas

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```
