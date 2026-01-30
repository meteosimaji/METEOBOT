# simajilord.com デプロイ（1行コマンド向け）

このリポジトリは、Bot本体（Discord）＋ /operator のWebパネル（aiohttp）を持っています。
公開は Nginx リバースプロキシで `simajilord.com -> 127.0.0.1:8080` に転送する想定です。

前提:
- DNS: simajilord.com と www.simajilord.com がサーバIPを向いている
- `.env` はリポジトリ直下（bot.py と同じ階層）に置く

## Nginx セットアップ（HTTPのみ）
以下を「上から1行ずつ」実行してください。

1) Apacheがいたら止める:
sudo systemctl disable --now apache2 || true

2) nginx を入れる:
sudo apt update
sudo apt install -y nginx

3) site設定を配置:
sudo cp deploy/nginx/simajilord.com.conf /etc/nginx/sites-available/simajilord.com

4) 壊れたリンクがあれば消して、有効化:
sudo rm -f /etc/nginx/sites-enabled/simajilord.com
sudo ln -s /etc/nginx/sites-available/simajilord.com /etc/nginx/sites-enabled/simajilord.com

5) テストして反映:
sudo nginx -t
sudo systemctl restart nginx

6) 確認:
curl -I http://simajilord.com/

## HTTPS（Let’s Encrypt）
certbot の対話があるので手動推奨:

sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d simajilord.com -d www.simajilord.com

## Botをbot.pyだけで起動
source venv/bin/activate
python bot.py

## systemdで常駐（任意）
※ `/operator` を headed (`ASK_OPERATOR_HEADLESS=false`) で使う場合、GUI の無いサーバでは Xvfb が必要です。
この repo の `deploy/systemd/capbot.service` は `xvfb-run` で bot をラップする想定なので、先に入れてください。

sudo apt update
sudo apt install -y xvfb xauth

sudo cp deploy/systemd/capbot.service /etc/systemd/system/capbot.service
sudo systemctl daemon-reload
sudo systemctl enable --now capbot
sudo systemctl status capbot --no-pager -l
