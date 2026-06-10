---
layout: two-cols
---

::title::
Colab演習の流れ

::left::
<span class="font-bold">学習</span>

1. 必要なライブラリをインストールする
2. Roboflowからデータセットを取得する
3. YOLOモデルを読み込む
4. `face` データセットで学習する
5. 学習結果を確認する

::right::
<span class="font-bold">推論</span>

1. テスト画像で推論する
2. 検出結果を画像に描画する
3. Webカメラで画像を撮影する
4. 撮影画像で推論する
5. `face` が検出されるか確認する

::conc::
データセット作成、学習、推論までを一つの流れとして体験する

---

::title::
ライブラリの準備

::default::
<Transform :scale="0.9">

```python
!pip install ultralytics roboflow
```

</Transform>

<br>

```python
from ultralytics import YOLO
from roboflow import Roboflow
```

<br>

- `ultralytics`: YOLOの学習と推論に使う
- `roboflow`: Roboflowからデータセットを取得する
- Colabでは初回実行時にインストールが必要

---

::title::
Roboflowからデータセットを取得

::default::
<Transform :scale="0.82">

```python
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
version = project.version(1)
dataset = version.download("yolov8")

print(dataset.location)
```

</Transform>

<br>

- `YOUR_API_KEY` はRoboflowのAPIキーに置き換える
- `YOUR_WORKSPACE` と `YOUR_PROJECT` はプロジェクトに合わせる
- ダウンロード後に `data.yaml` が作成される

---
layout: two-cols
---

::title::
YOLOで学習する

::left::
<span class="font-bold">小さいモデルから始める</span>

```python
model = YOLO("yolov8n.pt")
```

<br>

- `n` は nano の意味
- 軽くてColabで扱いやすい
- まず動く流れを確認する

::right::
<span class="font-bold">学習コード</span>

```python
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=30,
    imgsz=640,
    batch=16
)
```

<br>

画像枚数やColab環境に応じて、`epochs` や `batch` を調整する

::conc::
最初は高精度より、データセットからモデルを作る流れを完成させる

---

::title::
画像で推論する

::default::
<Transform :scale="0.9">

```python
best_model = YOLO("runs/detect/train/weights/best.pt")

results = best_model.predict(
    source="test_image.jpg",
    conf=0.25,
    save=True
)
```

</Transform>

<br>

- `best.pt` は学習中に保存された最良モデル
- `conf` は検出結果を表示する確信度のしきい値
- `save=True` にすると、検出結果の画像が保存される
