"""Запуск MCP-сервера SynthDataService.

Usage:
    python scripts/run_mcp.py            # stdio-транспорт (по умолчанию)
"""
import sys
from pathlib import Path

# Корень репозитория - чтобы импортировались src/mcp_server и synth_service
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mcp_server.server import mcp  # noqa: E402

if __name__ == "__main__":
    mcp.run()
