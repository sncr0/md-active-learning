"""Dev entry point for the dashboard API.

Usage:  uv run python scripts/run_api.py
Reads snapshots from ./data by default; override with MDAL_DATA_DIR.
"""

from __future__ import annotations

import uvicorn

if __name__ == "__main__":
    uvicorn.run("mdal.api.app:app", host="127.0.0.1", port=8000, reload=True)
