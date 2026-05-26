# 1. プロジェクト概要
- **目的**: 集団飼育環境下におけるマウスのひっかき行動（scratch bout）を、姿勢推定（SLEAP, YOLOv8-pose, DeepLabCut）および物理単位（mm/s, mm）のルールベース判定アルゴリズムによって自動検出する「学術品質パイプライン」と、インタラクティブに補正・再計算を行う「SPAダッシュボード」の統合システム。
- **重要な指標**: 
  - **学術的再現性と測定の信頼性**: 判定基準はすべてピクセルではなく物理単位（mm/s, mm）を用い、カメラのズレや撮影条件によらない普遍性を保証する。
  - **GUIとバックエンドの完全なロジック同期**: 手動補正時のGUI側再計算ロジック（`app.js` 内）と、パイプライン側ロジック（`feature_extraction.py` 内）が同一の閾値・演算式で動作し、一貫した結果を出力すること。
  - **プレミアムな操作性**: 骨格座標のドラッグ＆ドロップ補正、IDスワップ、バルクオフセット等の高度な操作を滑らかなアニメーションと直感的なUI/UXで提供する。

# 2. 技術スタック
- **フロントエンド**: 
  - HTML5 (`index.html`), Vanilla CSS (`index.css` - ダークモード、グラスモフィズム、プレミアムデザイン)
  - Vanilla JavaScript (`app.js`, `visualizer.js` (Canvas描画/ドラッグ操作), `charts.js` (Chart.jsグラフ), `constants.js`, `demoData.js`)
- **バックエンド**: 
  - Python 3.9+ (FastAPI / `app_server.py`)
- **姿勢推定 / 機械学習**:
  - SLEAP, YOLOv8-pose (Ultralytics), DeepLabCut (SuperAnimal / Model Zoo)
- **環境・パッケージ管理**:
  - Conda (`environment.yml`), pip (`requirements.txt`)
- **実行環境**:
  - Windowsローカル環境（`setup.bat`, `start.bat`）、Google Colab (`mouse_scratch_dashboard_colab.ipynb`)

# 3. コーディングルール
- **ファイル命名ルール**:
  - **Pythonスクリプト**: すべて「スネークケース（小文字とアンダースコア: `feature_extraction.py`）」とすること。
  - **JavaScriptスクリプト**: 原則として「キャメルケース（`demoData.js`）」または「小文字一単語（`app.js`, `charts.js`）」とすること。
- **変数名・関数名の命名ルール**:
  - **Python**: 「スネークケース（`scratch_prob`, `calculate_speed_mm_s`）」とすること。
  - **JavaScript**: 「キャメルケース（`recalculateFrameKinematics()`, `pixelToMmRatio`）」とすること。
- **コメント・ドキュメント**:
  - 複雑な物理計算（速度、距離、振動数等の判定ロジック）や、HSVカラーを用いたID識別処理など、学術検証に関わる箇所には、後で人間が検証できるように数式や意図を明記した **日本語のコメント** を必ず残すこと。
- **GUI ↔ バックエンド同期ルール**:
  - 特徴量抽出やスクラッチ判定のアルゴリズムを変更する際は、`feature_extraction.py` (Python) と `app.js` の `recalculateFrameKinematics` (JS) の **双方を同時に修正し、結果に乖離が生じないよう厳密に一致** させること。

# 4. よく使うコマンド
- **FastAPI サーバーの起動**: 
  - `python app_server.py` または `start.bat`
- **解析パイプラインの実行 (CLI)**:
  - `python pipeline.py --video <動画パス> --engine yolov8 --model_path models/mouse_pose_v2.pt --px-to-mm 0.231 --out_dir outputs`
- **精度評価 (手動GTとの照合)**:
  - `python evaluate.py --auto outputs/cage_01_frame_labels.csv --gt data/gt/cage_01_gt.csv --format frame --fps 30 --out_dir outputs`
- **テストの実行**:
  - Identity Swap 分割結合テスト: `python test_color_id.py`
  - 姿勢推定エンジン抽象化ユニットテスト: `python -m unittest test_pose_engine.py`

# 5. AIの振る舞いに関する重要ルール
- **既存スタイルの踏襲**: コードを修正・追加する際は、既存のクラス設計（`PoseEngine` 抽象化など）や、Python/JavaScriptそれぞれの命名規則・コード設計を忠実に真似してください。
- **学術的ロジックへの配慮**: 速度や距離、振動数の閾値判定ロジック（`config.py` や `feature_extraction.py`）に変更を加える場合は、学術的妥当性に影響を及ぼす可能性があるため、細心の注意を払い、変更内容とその影響について人間に明確に説明してください。
- **外部パッケージインストールの確認**: パッケージを新しく追加する場合は、PyTorchやCUDA、DeepLabCutなどの動作要件が極めてデリケートであるため、必ず実行前に人間に確認をとってください。
- **安全な推測の回避**: わからないパラメータ設計（校正値など）や不明な要件がある場合は、勝手に推測して決めつけず、「〇〇の仕様や定義について教えてください」と人間に直接質問して合意を得てください。
