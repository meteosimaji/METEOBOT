#!/usr/bin/env bash
set -euo pipefail

# HTTP only setup for nginx reverse proxy to operator server (127.0.0.1:8080).
# NOTE: certbot is interactive; run it manually after this script.

sudo systemctl disable --now apache2 || true

sudo apt update
sudo apt install -y nginx

sudo cp deploy/nginx/simajilord.com.conf /etc/nginx/sites-available/simajilord.com
sudo rm -f /etc/nginx/sites-enabled/simajilord.com
sudo ln -s /etc/nginx/sites-available/simajilord.com /etc/nginx/sites-enabled/simajilord.com

sudo nginx -t
sudo systemctl restart nginx

echo "OK: nginx reverse proxy is configured. Try: curl -I http://simajilord.com/"
