# ThreeFold runtime guide (LatentDiffusion-2DChestCT-CCIA24)

This document describes how to run this app on TFGrid/ThreeFold with the project Dockerfiles and flist flow.

## Dockerfiles for ThreeFold

- `Dockerfile.threefold` → GPU-oriented runtime
- `Dockerfile.threefold-cpu` → CPU-only runtime

Both are prepared for zinit-based runtime with nginx fronting the app.

## Runtime architecture

- Entrypoint: `/sbin/zinit init`
- zinit services:
  - `app` → runs `scripts/app-run.sh` (starts `inference_app.py`)
  - `nginx` → runs `scripts/nginx-run.sh`
- HTTP:
  - external UI on port `80` (nginx)
  - internal Flask app on `127.0.0.1:7860`

Nginx proxy config is in `nginx/default.conf`.

## Auth behavior

Auth is implemented inside `inference_app.py`:

- If `PASSWORD` is set and non-empty → UI is protected by login page (`/login`)
- If `PASSWORD` is missing/empty → no login protection
- `/logout` clears session

`FLASK_SECRET_KEY` is used for Flask session signing.

## Environment variables

Common runtime variables to set in deployment template:

- `ENABLE_SSH`
  - `"1"` to enable SSH
  - `"0"` or omit to keep SSH disabled
- `SSH_PUBLIC_KEY`
  - required if SSH is enabled
- `PASSWORD`
  - password for UI login
- `FLASK_SECRET_KEY`
  - session secret used by Flask

## Troubleshooting

### 502 Bad Gateway

Usually means nginx is up but app is not reachable on `7860`.

Check in VM shell:

```bash
zinit list
ss -lntp | grep 7860
```

If app is down, inspect app startup:

```bash
/opt/app/scripts/app-run.sh
```

### Login not shown

Ensure `PASSWORD` is present and non-empty in runtime environment.

### SSH not working

Ensure both:

- `ENABLE_SSH=1`
- `SSH_PUBLIC_KEY` set
