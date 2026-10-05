[[Japanese](README.md)/[English](README_EN.md)]

# cctagpy

https://github.com/user-attachments/assets/be13113f-e8fb-4436-af6c-928edb6b2e1b

[CCTag](https://github.com/alicevision/CCTag) のCPU検出パイプラインを pure-Python(NumPy/SciPy/Numba)に移植したものです。<br>
同心円フィデューシャルマーカーの検出・識別を行います。OpenCV/CUDAには依存しません。

# Features
- OpenCV/CUDAに依存しません(NumPy/SciPy/Numbaのみ)。Pillowは画像ファイルのデコードにだけ使います
- グレースケール変換を含め、アルゴリズムの全工程をNumPy/SciPy/Numbaで実装しています
- 重い処理はNumba JITとマルチコア並列化で高速化しています
- C++参照実装とは、ビット単位では一致しません(「Differences from C++ Version」参照)

# Purpose of This Repository
CPUパイプラインのPython移植と、NumPy/Numbaによる高速化を検証するリポジトリです。

# Requirements
```
Python 3.14 or later

numba          0.67.0    or later
numpy          2.5.3     or later
pillow         12.3.0    or later
scipy          1.18.1    or later
pytest         9.1.1     or later   # テスト用
opencv-python  4.9       or later   # Webカメラデモ用(cctagpy自体の依存ではありません)
```

# Installation
PyPIには公開していません。GitHubから直接インストールしてください。

```bash
pip install git+https://github.com/Kazuhito00/cctagpy-prototype.git
# uvの場合
uv add git+https://github.com/Kazuhito00/cctagpy-prototype.git
```

開発する場合は、クローンして依存を入れてください。
```bash
git clone https://github.com/Kazuhito00/cctagpy-prototype
cd cctagpy-prototype

# uvを使う場合(開発用依存も入ります)
uv sync

# pipを使う場合
pip install -e .
pip install "pytest>=9.1.1"
```

以降のコマンド例は `uv run` 付きで記載しています。pipで入れた環境では `uv run` を外して実行してください。

# Usage

### Quick Start
マーカー画像を生成して、検出します。
```bash
uv run python examples/demo_generate_marker.py --id 5
uv run python -m cctagpy -n 3 -i markers_out/marker_3crowns_0005.png
```

出力例です。左から、中心の x 座標、y 座標、マーカーID、status(1 なら信頼できる識別結果)です。
```text
#frame 0
Detected 1 candidates
419.91525528048277 419.754688594152 5 1
```

### CLI
```bash
uv run python -m cctagpy -n 3 -i path/to/image.png
```

検出したマーカーごとに1行 `x y id status` を出力します(C++版 `detection` サンプルアプリと同じ形式)。<br>
先頭に `#frame 0` と `Detected N candidates` の行が付きます。

- `-i, --input`: 入力画像のパス(必須)
- `-n, --nrings`: マーカーのリング数(3または4、デフォルト3)
- `--seed N`: RANSACの乱数シードを固定します
- `--fast-identification`: 識別段階の中心探索を、実験的な最適化版に切り替えます。高速ですが、C++参照実装とビット単位では一致しません

### Python
```python
import numpy as np
from cctagpy import Parameters, cctag_detection, load_gray_image

gray = load_gray_image("image.png")  # 高さ×幅の2次元 uint8 配列
markers = cctag_detection(gray, Parameters(n_crowns=3), rng=np.random.default_rng(0))
for m in markers:
    print(m.x(), m.y(), m.id, m.status)  # status == 1 なら信頼できる識別結果
```

`x` は横方向、`y` は縦方向の画素座標です。マーカーIDは0始まりです。

### Demo
Webカメラデモには OpenCV が必要です。
```bash
uv pip install -r examples/requirements.txt
```

Webカメラに960x540の解像度を要求し、各フレームの検出結果を重畳表示します。
```bash
uv run python examples/demo_webcam.py
```

印刷用マーカー画像(PNG)を `markers_out/` に生成します(`--id`/`--n-crowns`/`--all` など)。
```bash
uv run python examples/demo_generate_marker.py
```

# Test
```bash
uv run pytest
```

合成した同心円マーカー画像に対する、各処理段階と検出・識別のテストです。C++参照実装の出力との比較テストは
含めていません。

# Performance

NumPyの配列演算を基本とし、配列全体の一時確保がボトルネックだった処理(ピラミッドリサイズ、Cannyエッジ検出、
細線化、識別段階のホモグラフィ再サンプリング、楕円growingのフラッドフィル、RANSACの候補抽出・スコアリングなど)は
Numba JITで実装しています。

主な最適化は次のとおりです。

- Pythonループ(`is_another_segment`、`vote.outlier_removal`のスコアリング等)をバッチNumPy呼び出しに変更
- Numba `prange` によるマルチコア並列化(ピラミッド構築、レイマーチング、識別の再サンプリングなど)
- RANSACの5点/8点フィットをLAPACK経由ではなく直接実装(Gaussian消去・解析的固有値分解)に変更
- 乱数抽選(`rng.choice`の大量呼び出し)を`rng.permuted`/rejection samplingベースの一括抽選に変更
- レベル0への再フィット時に使う楕円ハル内の点選択(`detection.select_edge_point_in_elliptic_hull`)を
  スキャンライン走査ごとNumba化
- `geometry.Ellipse`のconic行列計算(`compute_matrix`、3x3逆行列)をNumba化。`Ellipse`はRANSAC試行・
  growingの各ステップ・ハル計算などで繰り返し構築します

RANSACの乱数抽選と数値ソルバー、および楕円のconic行列計算は、最適化前の実装と結果がビット単位で異なる場合があります
(Gaussian消去や解析的固有値分解は、LAPACKと丸め方が異なります)。conic行列計算の変更については、
8シード×両サンプル画像で最適化前と比較しました。マーカーID集合は一致し、中心座標の平均ずれは最大0.367px
(該当マーカー自身のシード間ばらつきは2.5〜2.9px)、他のマーカーは0.02px以下でした。

14コアのマシンで1920x1440のサンプル画像を処理すると、NumPyのみの実装は約2.2秒、現在の実装は約0.15〜0.20秒で、
約11〜15倍になります。

入力画像を縮小すると、処理時間はほぼ画素数に比例して減ります。一方、マーカーのリング幅が画素単位で
細くなりすぎると、再現率が画像依存で非線形に低下します。安全な縮小率は画像ごとに異なるため、本ライブラリは
縮小を行いません。速度が必要な場合は、自分の画像で再現率を測定した上で、呼び出し側で縮小してから
`cctag_detection` に渡してください。

`python -m cctagpy` の実行時間には、`numba` のimport(約0.3秒)と、初回のJITコンパイルまたはキャッシュの読み込みも
含まれます。画像を連続して処理する場合は、同じプロセスで `cctag_detection()` を繰り返し呼び出してください。

# Differences from C++ Version

- ビット単位では一致しません。RANSACの候補抽出はNumPyのRNGを使うため、C++版のPCG32ストリームとは異なります。
  画像ピラミッドのbilinearリサイズは、C++版の`cv::resize`と丸め方が異なる場合があります。`Ellipse`の
  conic行列計算も、Numba/NumPyで浮動小数点の計算順序が異なります(「Performance」参照)。
  検証はビット単位ではなく、許容誤差ベース(マーカーID・中心座標の一致)で行っています。
- CUDAパイプラインには対応していません(CPUのみの再実装です)。
- C++側のデッドコード(`SubPixEdgeOptimizer`、未使用のカット選択コスト関数、`conditionerFromImage`)は
  移植していません。
- C++版の次の挙動は、そのまま移植しています。ピラミッドの上位レベルで検出したマーカーは、レベル0での再フィットに
  失敗すると `x()`/`y()` がそのレベルの座標のまま残ります。`n_circles` は、リング数によらず3リング相当の値になります。
- 参照実装との出力を比較できるように、C++版の既知の挙動(`CCTag::_idSet`が識別後に常に空になる)に合わせて、
  識別後の`id_set`は空のままにしています。

# Project Structure

```text
README.md                # README（日本語）
README_EN.md             # README（英語）
LICENSE                  # MPL-2.0
pyproject.toml           # パッケージ定義
src/cctagpy/
  detection.py           # 検出のトップレベル (cctag_detection)
  identification.py      # マーカーバンクに対する識別
  ...                    # canny/thinning/vote/ransac/ellipse_growing 等 (各docstringにC++対応ファイルを記載)
examples/                # Webカメラデモ・マーカー画像生成
tests/                   # pytest用テストケース
```

# Author
高橋かずひと(https://x.com/KzhtTkhs)

# License
cctagpy is under [Mozilla Public License 2.0](LICENSE)(移植元のCCTagと同じ)。<br>
