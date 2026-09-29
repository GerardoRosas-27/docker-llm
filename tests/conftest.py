import os
import tempfile

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="obrador-test-")
os.environ["AUTO_DOWNLOAD_DEFAULT"] = "0"
os.environ["ADMIN_TOKEN"] = ""
os.environ["API_KEY"] = ""
os.environ["MAX_MODEL_BYTES"] = "9000000000"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def clean_db():
    db.init()
    db.reset()
    yield
    db.reset()
