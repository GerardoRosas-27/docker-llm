import os
import tempfile

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="obrador-test-")
os.environ["AUTO_DOWNLOAD_DEFAULT"] = "0"
os.environ.pop("ADMIN_TOKEN", None)
os.environ.pop("API_KEY", None)
os.environ["MAX_MODEL_BYTES"] = "9000000000"
# Solo para las pruebas: sin MASTER_SECRET todo queda cerrado (fail-closed).
TEST_SECRET = "secreto-de-pruebas-" + "z" * 40
os.environ["MASTER_SECRET"] = TEST_SECRET

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import access, db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def _test_client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def anon(_test_client, monkeypatch):
    """Cliente sin credenciales."""
    monkeypatch.setenv("MASTER_SECRET", TEST_SECRET)
    _test_client.headers.pop("Authorization", None)
    yield _test_client
    _test_client.headers.pop("Authorization", None)


@pytest.fixture
def client(anon):
    """Cliente con una sesión de administración válida (como el panel tras entrar)."""
    anon.headers["Authorization"] = f"Bearer {access.issue_session()['token']}"
    yield anon


@pytest.fixture(autouse=True)
def clean_db():
    db.init()
    db.reset()
    access.limiter.reset()
    yield
    db.reset()
    access.limiter.reset()
