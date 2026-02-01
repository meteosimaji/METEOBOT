# Cookie JSON → cookies.txt ガイド

このフォルダは、ブラウザ拡張などが出力する **JSON 形式の Cookie** を、yt-dlp が読める
**Netscape 形式 cookies.txt** に変換するためのものです。

> ⚠️ Cookie はログイン情報そのものです。共有しないでください。
> リポジトリに追加しないように、ファイルは必ず **git 管理外** に置いてください。

## 使い方

1. JSON を保存する  
   例: `cookie/cookies.json` を開いて値を貼り替えます（ファイル名変更は不要）。

2. 変換スクリプトを実行する（または Bot 起動時に自動生成）

   ```bash
   python cookie/convert_json.py cookies.json -o cookies.txt
   chmod 600 cookies.txt
   ```

   Bot を起動すると `cookie/cookies.json` がある場合は自動で `cookie/cookies.txt` を作成します。

3. BOT を起動するときに環境変数を指定する

   ```bash
   export SAVEVIDEO_COOKIES_AUTO=false
   export SAVEVIDEO_COOKIES_FILE=/path/to/cookies.txt
   ```

## JSON 形式の注意点

以下のいずれかの形で受け付けます:

```json
[
  {"domain": ".x.com", "name": "auth_token", "value": "..."}
]
```

```json
{"cookies": [
  {"domain": ".x.com", "name": "auth_token", "value": "..."}
]}
```

```json
{
  "x": [{"domain": ".x.com", "name": "auth_token", "value": "..."}],
  "tiktok": [{"domain": ".tiktok.com", "name": "sessionid", "value": "..."}]
}
```

## セキュリティの注意

* `auth_token`, `sessionid` などは **アカウントを操作できるキー** です。
* 共有や公開は絶対に避けてください。
* もし漏洩の可能性がある場合は、各サービス側でログアウト/セッション無効化を行ってください。
