#!/usr/bin/env bash
# =====================
# Guarvi1 VPS setup - Groww API live-data bot on a static-IP VM
# Target: Ubuntu 22.04/24.04 (Oracle Cloud Always Free works out of the box).
#
# What it does:
#   1. Installs system packages (python venv, git, ufw)
#   2. Configures ufw: SSH rate-limited in, all outbound open (Groww needs
#      only outbound access)
#   3. Creates .venv and installs requirements.txt
#   4. Provisions .env from .env.example (chmod 600) - YOU add the credentials
#   5. Installs + enables the systemd service 'guarvi-live-data'
#   6. Prints the VM's public IP -> whitelist it in the Groww developer portal
#
# Usage (from the project root):
#   sudo bash scripts/setup_vps.sh
#   sudo bash scripts/setup_vps.sh --symbols "RELIANCE TCS" --poll-seconds 60
#   sudo bash scripts/setup_vps.sh --domain signal.example.com   # enable HTTPS
#   sudo bash scripts/setup_vps.sh --skip-firewall --no-service --no-web
# =====================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="guarvi-live-data"
RUN_USER="${SUDO_USER:-$(id -un)}"
SKIP_FIREWALL=0
NO_SERVICE=0
NO_WEB=0
UNIVERSE="nifty100"
SYMBOLS=""
CANDLE_INTERVAL="5minute"
POLL_SECONDS=300
WEB_DOMAIN=""

usage() { grep '^# ' "$0"; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-firewall) SKIP_FIREWALL=1 ;;
    --no-service) NO_SERVICE=1 ;;
    --universe) UNIVERSE="$2"; shift ;;
    --symbols) SYMBOLS="$2"; shift ;;
    --candle-interval) CANDLE_INTERVAL="$2"; shift ;;
    --poll-seconds) POLL_SECONDS="$2"; shift ;;
    --no-web) NO_WEB=1 ;;
    --domain) WEB_DOMAIN="$2"; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

echo "==> Project dir : $PROJECT_DIR"
echo "==> Service user: $RUN_USER"

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: run with sudo: sudo bash scripts/setup_vps.sh" >&2
  exit 1
fi

# --- 1. System packages -----------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
echo "==> Installing system packages..."
apt-get update -y
apt-get install -y python3-venv python3-pip git ufw

# --- 2. Firewall (outbound must stay open for the Groww API) ----------------
if [[ $SKIP_FIREWALL -eq 0 ]]; then
  SSH_PORT="$(grep -E '^[[:space:]]*Port[[:space:]]+' /etc/ssh/sshd_config 2>/dev/null | tail -1 | awk '{print $2}')"
  SSH_PORT="${SSH_PORT:-22}"
  echo "==> ufw: SSH on port ${SSH_PORT} (rate-limited); deny inbound, allow outbound"
  ufw allow "${SSH_PORT}/tcp" >/dev/null
  ufw limit "${SSH_PORT}/tcp" >/dev/null 2>&1 || true
  ufw default deny incoming
  ufw default allow outgoing
  ufw --force enable
  ufw status verbose
else
  echo "==> Skipping firewall setup (--skip-firewall)"
fi

# --- 3. Python virtual environment ------------------------------------------
echo "==> Creating .venv and installing Python dependencies..."
python3 -m venv "$PROJECT_DIR/.venv"
"$PROJECT_DIR/.venv/bin/pip" install --upgrade pip >/dev/null
"$PROJECT_DIR/.venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

# --- 4. .env -----------------------------------------------------------------
if [[ ! -f "$PROJECT_DIR/.env" ]]; then
  cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
  echo "==> Created .env from .env.example"
fi
chmod 600 "$PROJECT_DIR/.env"
chown -R "$RUN_USER":"$RUN_USER" "$PROJECT_DIR"

# --- 5. Data directories -----------------------------------------------------
mkdir -p "$PROJECT_DIR/data/live/nse"
chown -R "$RUN_USER":"$RUN_USER" "$PROJECT_DIR"

# --- 6. systemd service ------------------------------------------------------
RUNNER_ARGS="--candle-interval ${CANDLE_INTERVAL} --poll-seconds ${POLL_SECONDS}"
if [[ -n "$SYMBOLS" ]]; then
  RUNNER_ARGS="--symbols ${SYMBOLS} ${RUNNER_ARGS}"
else
  RUNNER_ARGS="--universe ${UNIVERSE} ${RUNNER_ARGS}"
fi

if [[ $NO_SERVICE -eq 0 ]]; then
  cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=Guarvi1 live market data fetcher (Groww API)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${PROJECT_DIR}
EnvironmentFile=${PROJECT_DIR}/.env
ExecStart=${PROJECT_DIR}/.venv/bin/python scripts/live_data_runner.py ${RUNNER_ARGS}
Restart=always
RestartSec=30
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable "${SERVICE_NAME}.service"
  echo "==> Installed systemd service '${SERVICE_NAME}' (enabled, NOT started yet)"
fi

# --- 6b. Web portal (read-only FastAPI dashboard) ---------------------------
if [[ $NO_WEB -eq 0 ]]; then
  echo "==> Installing web portal (nginx + FastAPI)..."
  apt-get install -y nginx
  WEB_SERVICE="guarvi-web"
  SIGNAL_SERVICE="guarvi-signal"

  cat > "/etc/systemd/system/${WEB_SERVICE}.service" <<EOF
[Unit]
Description=Guarvi1 web portal (read-only FastAPI)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${PROJECT_DIR}
EnvironmentFile=${PROJECT_DIR}/.env
ExecStart=${PROJECT_DIR}/.venv/bin/uvicorn web.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=10
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

  # Nightly signal rebuild (after market close) so the portal serves fresh ranks.
  # The pipeline runs in two stages so that the web portal has fresh technical
  # indicators (RSI, MACD, returns, vol) every day, not just a refreshed close:
  #   1. ExecStartPre runs refresh_live_features, which aggregates the day's
  #      5-min Groww candles into a daily row, runs the feature pipeline over
  #      the warmup-aware full series, and appends new rows to
  #      nifty100_features.parquet (idempotent).
  #   2. ExecStart runs build_signal with --refresh-from-live, which overlays
  #      the latest intraday close onto the scored output for the freshness
  #      pill in the UI.
  cat > "/etc/systemd/system/${SIGNAL_SERVICE}.service" <<EOF
[Unit]
Description=Guarvi1 nightly signal rebuild
After=network-online.target

[Service]
Type=oneshot
User=${RUN_USER}
WorkingDirectory=${PROJECT_DIR}
EnvironmentFile=${PROJECT_DIR}/.env
ExecStartPre=${PROJECT_DIR}/.venv/bin/python -m scripts.refresh_live_features
ExecStart=${PROJECT_DIR}/.venv/bin/python -m scripts.build_signal --refresh-from-live
EOF

  cat > "/etc/systemd/system/${SIGNAL_SERVICE}.timer" <<EOF
[Unit]
Description=Guarvi1 nightly signal rebuild timer

[Timer]
OnCalendar=*-*-* 16:00:00 Asia/Kolkata
Persistent=true

[Install]
WantedBy=timers.target
EOF

  # nginx site (proxies /api to uvicorn, serves dashboard via FastAPI)
  NGINX_SITE="/etc/nginx/sites-available/guarvi"
  if [[ -n "$WEB_DOMAIN" ]]; then
    SERVER_NAME="$WEB_DOMAIN"
    CAT_CMD="certbot --nginx -d $WEB_DOMAIN || true"
  else
    SERVER_NAME="_"
    CAT_CMD="echo 'No --domain given: serving HTTP only on :80 (add TLS later with certbot)'"
  fi
  cat > "$NGINX_SITE" <<EOF
server {
    listen 80;
    listen [::]:80;
    server_name ${SERVER_NAME};

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
    }
}
EOF
  ln -sf "$NGINX_SITE" /etc/nginx/sites-enabled/guarvi
  rm -f /etc/nginx/sites-enabled/default
  nginx -t && systemctl reload nginx

  systemctl daemon-reload
  systemctl enable "${WEB_SERVICE}.service"
  systemctl enable "${SIGNAL_SERVICE}.timer"
  systemctl start "${WEB_SERVICE}.service"
  systemctl start "${SIGNAL_SERVICE}.timer"
  echo "==> Web portal installed: systemd '${WEB_SERVICE}', timer '${SIGNAL_SERVICE}'"
  echo "==> nginx site at $NGINX_SITE (server_name=${SERVER_NAME})"
  eval "$CAT_CMD"
fi

# --- 7. Public IP for Groww whitelisting -------------------------------------
echo
echo "==> Detecting this VM's public IP..."
"$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/check_static_ip.py" || true

echo
echo "================ NEXT STEPS ================"
echo "1. Whitelist the public IP above in the Groww developer portal."
echo "2. Add credentials:  nano ${PROJECT_DIR}/.env"
echo "   (GROWW_API_KEY, GROWW_SECRET)"
echo "3. Verify connectivity:"
echo "   cd ${PROJECT_DIR} && .venv/bin/python scripts/fetch_live_data.py --symbols RELIANCE --quote-only"
if [[ $NO_SERVICE -eq 0 ]]; then
  echo "4. Start the data daemon:"
  echo "   sudo systemctl start ${SERVICE_NAME}"
  echo "   journalctl -u ${SERVICE_NAME} -f"
  echo "   (stop: sudo systemctl stop ${SERVICE_NAME})"
fi
if [[ $NO_WEB -eq 0 ]]; then
  echo "5. Web portal is live (nginx -> uvicorn :8000). Open http://<this-VM-ip>/"
  echo "   Rebuild signal now: sudo systemctl start guarvi-signal"
  echo "   Watch: journalctl -u guarvi-web -f"
  if [[ -z "$WEB_DOMAIN" ]]; then
    echo "   For HTTPS: re-run with --domain your.domain.com (certbot handles TLS)."
  fi
fi
echo
echo "NOTE (Oracle Cloud): images ship default iptables REJECT rules for inbound"
echo "traffic. Outbound (what Groww needs) is allowed by default; the ufw rules"
echo "above manage inbound if you later expose extra ports."
echo "============================================"
