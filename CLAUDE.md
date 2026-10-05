# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`cctagpy` は [CCTag](https://github.com/alicevision/CCTag)(同心円フィデューシャルマーカーの検出・識別ライブラリ)のCPU検出パイプラインを、pure-Python(NumPy/SciPy/Numba)に移植したもの。OpenCV/CUDAには依存しない。Pillowは画像ファイル(PNG/JPEG)をピクセル配列にデコードする用途のみに使い、グレースケール変換を含むアルゴリズム全体はNumPy/SciPy/Numbaで直接実装している。

このリポジトリは `alicevision/CCTag` のC++ CPUパイプラインをモジュール単位で1対1に近い形で移植したもの。**各モジュールのdocstringには対応するC++ファイル(例: `src/cctag/Detection.cpp`)と、挙動上の注意点が明記されている** — 実装を変更する前に必ずそのdocstringを読むこと。C++版のソースはこのリポジトリには同梱されていない。

## コマンド

```bash
uv sync                    # インストール(Numbaは必須依存として同梱)
uv run python -m cctagpy -n 3 -i path/to/image.png   # CLI実行
uv run pytest               # 単体テスト
uv run pytest tests/test_detection.py::test_name -v   # 単体テストを1つだけ実行
```

デモ(`examples/`。`opencv-python`が必要、`examples/requirements.txt`経由。cctagpy自体の依存ではない):

```bash
pip install -r examples/requirements.txt
python examples/demo_webcam.py            # Webカメラでのリアルタイム検出
python examples/demo_generate_marker.py   # 印刷用マーカーPNG生成
```

## アーキテクチャ

パイプラインは `src/cctagpy/detection.py` の `cctag_detection()` がトップレベルのエントリポイント。処理は大きく2段に分かれる:

1. **検出(Detection)** — `id == -1`, `status == 0` のマーカー候補を作る段階。マルチレゾリューション駆動(`pyramid.py` の `ImagePyramid`)で、レベルごとに以下の3ループを回す(C++の `Detection.cpp`/`Multiresolution.cpp` に対応):
   - Loop 1: `construct_flow_component_from_seed` — シードからエッジセグメントを構築
   - Loop 2: `complete_flow_component` — アウトライヤ除去(`outlier_removal`, RANSACベース)、楕円growing(`ellipse_growing.py`)
   - Loop 3: `cctag_detection` 内のマージ/重複除去 + レベル0でのアウター楕円再フィット
2. **識別(Identification)** — `identification.py` が候補に実際の `id`/`status` を割り当てる段階。マーカーバンク(`markers_bank.py`、各マーカーのリング半径比シグネチャ)と照合する。`status == 1` が信頼できる識別結果。

主要モジュールの対応関係(C++ファイル名はdocstring参照):

- `canny.py` / `thinning.py` — エッジ/勾配検出、細線化
- `vote.py` — ボーティング、エッジリンキング(`edge_linking`)、アウトライヤ除去
- `ransac.py` — RANSACによる楕円候補フィット(5点/8点、LAPACK不使用の直接実装)
- `ellipse_growing.py` / `fitting.py` / `geometry.py` — 楕円growing/フィッティング、`Ellipse`/`Circle`
- `edge_collection.py` — `EdgePointCollection`(全エッジ点の状態を保持する中心的なデータ構造)
- `identification.py` — ホモグラフィ再サンプリング、中心探索、識別スコアリング
- `params.py` — `Parameters`(`Params.hpp` の `kDefault*` を直接転記。CUDA専用フィールドは省略)
- `cctag.py` — `CCTag` マーカー型(`x()`, `y()`, `id`, `status`)
- `cli.py` — C++版 `detection` サンプルアプリと同じ出力形式(`x y id status`)のCLI

### Numba dispatch パターン

`_numba_utils.py` の `HAS_NUMBA` フラグで、パイプラインの重い箇所(ピラミッドリサイズ、Canny、細線化、識別の再サンプリング、楕円growingのフラッドフィル、RANSACの候補抽出・スコアリングなど)はNumba JIT実装とNumPy実装を両方保持し、`HAS_NUMBA` で分岐する。**フォールバック用の no-op `njit` デコレータは意図的に存在しない** — 明示ピクセルループのNumba版を素のPythonループとして動かすと、置き換え元のNumPyベクトル化版より大幅に遅くなるため。新しいステージをNumba化する際もこのdispatchパターン(両実装を残し、テストでビット一致をクロスチェックする)に従うこと。

### C++版との既知の差異(ビット完全一致ではない箇所)

- RANSACの候補抽出はNumPyのRNGを使用(C++版のPCG32ストリームとは異なる)
- 画像ピラミッドのbilinearリサイズは `cv::resize` と丸め方がわずかに異なる場合がある
- `Ellipse` のconic行列計算(`geometry.py`)はNumba化されており、`np.linalg.inv` とLAPACKで浮動小数点の計算順序がわずかに異なる
- C++側のデッドコード(`SubPixEdgeOptimizer`、未使用のカット選択コスト関数、`conditionerFromImage`)は意図的に移植していない
- C++版の既知のバグ(`CCTag::_idSet` が識別後に常に空になる)は、参照実装との出力比較を可能にするためあえて修正せず忠実に再現している

C++参照実装との比較は、ビット単位ではなく許容誤差ベース(マーカーID集合の一致、中心座標の差が数ピクセル以内)で行う。リポジトリ内にはC++参照出力との比較テストは含めていない。

### パフォーマンス上の注意

- GPU化(`numba.cuda`)は試作の上で見送っている。識別段階のホモグラフィ再サンプリングはGPUの方が遅く(転送コストが支配的)、Cannyは約2.5〜3倍速いが、主用途(960x540のリアルタイム映像)ではEnd-to-end全体への寄与が約2%と小さい。
- 入力画像の縮小は呼び出し側の責任(デフォルトでは行わない)。安全な縮小率は画像依存で非線形に再現率が落ちるため、ライブラリ側で決め打ちしない。
- `numba` のimportコスト(LLVM初期化、約0.3秒)があるため、`python -m cctagpy` を画像1枚ごとに1プロセス起動する使い方では高速化のメリットが相殺される。`cctag_detection()` を同一プロセス内で繰り返し呼ぶ用途(バッチ/サービス/動画ループ)で効果を発揮する設計。
