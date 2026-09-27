import os
import tempfile

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="obrador-test-")
os.environ["AUTO_DOWNLOAD_DEFAULT"] = "0"
os.environ["ADMIN_TOKEN"] = ""
os.environ["API_KEY"] = ""
os.environ["MAX_MODEL_BYTES"] = "9000000000"
