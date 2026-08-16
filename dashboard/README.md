# dashboard

React + Vite frontend for the md-active-learning campaign dashboard: a DOE view of which
simulations are complete vs. still running, with expandable per-run detail.

Talks only to the read-only API in `../src/mdal/api` — never to a store's `.duckdb` file
directly (see the repo README's "Dashboard" section for why).

```bash
npm install
npm run dev     # http://localhost:5173, proxies /api -> http://127.0.0.1:8000
```

Requires the API running separately: `uv run python scripts/run_api.py` from the repo root.
