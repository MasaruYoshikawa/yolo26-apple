# Apple Counter Web Application / りんご個数カウント・ウェブアプリケーション

このディレクトリは、YOLOv8物体検出モデルを利用して画像内のりんごの検出およびカウントを行うウェブベースのダッシュボードアプリケーションです。ドラッグ＆ドロップによる画像アップロード、対話的な前後比較スライダー、確信度（Confidence）やIoUしきい値のリアルタイム調整、および解析データのCSVエクスポート機能を備えています。

This folder contains a web-based Apple Detection & Counting dashboard leveraging the YOLOv8 model. It features drag-and-drop batch upload, an interactive original-vs-annotated slider, real-time threshold configuration, and CSV report export.

---

## 🌟 主な機能 (Key Features)

1. **ドラッグ＆ドロップ・バッチアップロード (Drag & Drop Batch Upload)**:
   - 複数の画像をまとめてドラッグ＆ドロップするだけで、非同期で順次解析が行われます。
2. **対話型比較スライダー (Interactive Before/After Slider)**:
   - 解析結果画像を元の画像と重ね合わせ、スライダーハンドルを動かして検出結果（バウンディングボックス）の有無を視覚的に比較できます。
3. **推論パラメータのリアルタイム調整 (Real-time Configuration)**:
   - 確信度閾値（Confidence Threshold）や非最大値抑制（IoU NMS Threshold）のスライダー、および推論サイズを操作し、検出精度を動的に微調整できます。
4. **解析履歴とCSV出力 (History Log & CSV Export)**:
   - セッション中に処理したすべての画像、カウント、推論時間を一覧表にし、結果のCSVファイルをワンクリックでダウンロードできます（Excel用の日本語文字化け対策BOM付き）。
5. **プレミアム・ダークテーマ (Premium Glassmorphism UI)**:
   - モダンなグラスモルフィズムスタイルを取り入れた流麗なデザインです。

---

## ⚙️ セットアップ & 実行方法 (Setup & Running)

### 1. 依存関係のインストール (Install Dependencies)
Python環境で以下のコマンドを実行し、必要なライブラリをインストールします。
Run the following command to install the required libraries:

```bash
pip install -r requirements.txt
```

> [!NOTE]
> 既にメインプロジェクトフォルダで `ultralytics`, `opencv-python`, `torch` 等がインストールされている場合は、追加で `flask` のみをインストールするだけで動作します。
> If you already have `ultralytics`, `opencv-python`, etc., installed, you only need to install `flask`:
> `pip install flask`

### 2. サーバーの起動 (Run the Server)
`web_app` フォルダ内で以下のコマンドを実行してウェブサーバーを起動します。
Execute the server script in the `web_app` directory:

```bash
python app.py
```

### 3. ブラウザでアクセス (Access via Browser)
サーバーが起動したら、ブラウザを開いて以下のURLにアクセスします。
Open your web browser and navigate to:

👉 **[http://localhost:5000](http://localhost:5000)**

---

## 📁 フォルダ構成 (Project Structure)

- `app.py`: Flaskバックエンドサーバー。画像推論処理およびAPIのエンドポイント。
- `requirements.txt`: 必要なパッケージ定義。
- `templates/index.html`: ダッシュボードのHTML構造。
- `static/css/style.css`: UIのスタイル定義（CSS変数とグラスモルフィズム効果）。
- `static/js/app.js`: 非同期アップロード、スライダー制御、CSV生成等のクライアントサイド制御ロジック。
