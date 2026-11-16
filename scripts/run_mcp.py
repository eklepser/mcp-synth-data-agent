import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mcp_server.server import mcp  

if __name__ == "__main__":
    mcp.run()
