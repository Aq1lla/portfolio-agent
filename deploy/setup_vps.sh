#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Portfolio Agent — VPS Deployment skripta
#
# Pretpostavke:
#   - Ubuntu 22.04+ / Debian 12+
#   - Root ili sudo pristup
#   - Projekt je kloniran u /home/portfolio/portfolio-agent
#
# Pokretanje:
#   chmod +x deploy/setup_vps.sh
#   sudo bash deploy/setup_vps.sh
# ══════════════════════════════════════════════════════════════

set -euo pipefail

APP_USER="portfolio"
APP_DIR="/home/${APP_USER}/portfolio-agent"
PYTHON_VERSION="3.12"

echo "═══════════════════════════════════════════════"
echo "  Portfolio Agent — VPS Setup"
echo "═══════════════════════════════════════════════"

# ── 1. Sistemske dependencije ──
echo ""
echo "[1/6] Instaliram sistemske dependencije..."
apt-get update -qq
apt-get install -y -qq python${PYTHON_VERSION} python${PYTHON_VERSION}-venv git

# ── 2. Korisnik ──
echo "[2/6] Postavljam korisnika..."
if ! id "${APP_USER}" &>/dev/null; then
    useradd -m -s /bin/bash "${APP_USER}"
    echo "  Korisnik '${APP_USER}' kreiran."
else
    echo "  Korisnik '${APP_USER}' već postoji."
fi

# ── 3. Virtual environment ──
echo "[3/6] Postavljam Python okruženje..."
sudo -u "${APP_USER}" bash -c "
    cd ${APP_DIR}
    python${PYTHON_VERSION} -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    echo '  Dependencije instalirane.'
"

# ── 4. Direktoriji ──
echo "[4/6] Kreiram direktorije..."
sudo -u "${APP_USER}" mkdir -p "${APP_DIR}/data" "${APP_DIR}/logs"

# ── 5. Systemd servis ──
echo "[5/6] Instaliram systemd servis..."
cp "${APP_DIR}/deploy/portfolio-agent.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable portfolio-agent
echo "  Servis instaliran i omogućen."

# ── 6. Provjera ──
echo "[6/6] Provjera..."
sudo -u "${APP_USER}" bash -c "
    cd ${APP_DIR}
    source .venv/bin/activate
    python scripts/setup_db.py
"

echo ""
echo "═══════════════════════════════════════════════"
echo "  Setup završen!"
echo ""
echo "  Sljedeći koraci:"
echo "  1. Postavi API ključeve:"
echo "     nano ${APP_DIR}/config/secrets.yaml"
echo ""
echo "  2. Pokreni agenta:"
echo "     sudo systemctl start portfolio-agent"
echo ""
echo "  3. Provjeri status:"
echo "     sudo systemctl status portfolio-agent"
echo "     journalctl -u portfolio-agent -f"
echo ""
echo "  4. Za jednokratni test:"
echo "     cd ${APP_DIR} && source .venv/bin/activate"
echo "     python runner.py --once"
echo "═══════════════════════════════════════════════"
