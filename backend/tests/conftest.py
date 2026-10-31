import sys
from pathlib import Path

# Backend modules use flat imports (``from config import …``).
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
