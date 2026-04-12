# Genetic Algorithm Workbench

uv monorepo with binary and real-valued GA variants, React frontend with 3D surface visualization.

## Quick start

```bash
./scripts/dev.sh
```

Opens at `http://127.0.0.1:5000`.

Requires [uv](https://docs.astral.sh/uv/) and [Node.js](https://nodejs.org/).

## Manual start

```bash
uv sync --all-packages
cd app/frontend && npm install && npm run build && cd ../..
uv run ga-app
```

## Development (hot reload)

```bash
uv run ga-app                          # API on :5000
cd app/frontend && npm run dev         # React on :5173 (proxies /api to Flask)
```

## Structure

```
packages/ga-core/     shared: abstract GA base, config, registry, objective functions
projects/p1-binary/   binary chromosome GA
projects/p2-real/     real-valued chromosome GA
projects/p3-parallel/ placeholder
app/                  Flask API + React frontend
```
