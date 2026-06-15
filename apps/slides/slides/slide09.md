---
theme: tut
layout: cover
---

::title::
CNNを用いた画像認識の仕組みと実装

::date::
2026年06月15日

::default::
山中春輝

---
layout: two-cols
---

::title::
授業計画

::left::
- 第1回 ガイダンス / 犬猫の分類を試してみよう
- 第2回 画像とは / 畳み込みの実装
- 第3回 プーリング層の実装 / 活性化関数とは
- 第4回 全結合層の実装 / ネットワークの構築
- 第5回 損失関数と最適化 / MNISTで学習と推論
- 第6回 犬猫分類の学習と精度評価
- 第7回 ResNetと転移学習による精度改善
- 第8回 物体検出タスクの基礎及び推論

::right::
- <span class="font-bold text-blue-500">第9回 物体検出モデルの精度評価</span>
- 第10回 領域分割タスクの基礎及び推論
- 第11回 データ拡張の重要性
- 第12回 エラー分析手法
- 第13回 総合演習（１）
- 第14回 総合演習（２）
- 第15回 総合演習（３）
- 第16回 総合演習（４）

::conc::
授業計画は変更される可能性があります

---
layout: toc
---

::title::
目次

---
layout: section
---

# 前回の復習

---

::title::
前回の復習

::default::
1. <span class="font-bold">物体検出</span>

    → 画像の中の「どこに」「何が」あるかを予測するタスク

2. <span class="font-bold">Roboflow</span>

    → 画像、クラス、バウンディングボックスを管理し、学習用データセットとして出力した

3. <span class="font-bold">YOLOによる学習</span>

    → `ultralytics` を使い、Roboflowのデータセットから検出モデルを作成した

4. <span class="font-bold">今回学ぶこと</span>

    → YOLOではなく<span class="text-blue-500 font-bold">torchvisionの事前学習済み重み</span>を使い、検出モデルを転移学習する

---
layout: section
---

# YOLOからtorchvisionへ

---
layout: two-cols
---

::title::
今回の方針

::left::
<span class="font-bold">前回</span>

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="data.yaml")
```

<br>

- YOLO専用のAPIを使う
- 学習ループはライブラリ側に任せる
- すぐに検出を体験しやすい

::right::
<span class="font-bold">今回</span>

```python
from torchvision import models
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights
)
```

<br>

- PyTorchのモデルを直接使う
- 出力層を自分で差し替える
- 損失、予測、評価を理解しやすい

::conc::
便利なAPIから一歩進み、検出モデルの中身と評価方法を理解する

---

::title::
torchvisionの検出モデル

::default::
<Transform :scale="0.9">

```python
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
```

</Transform>

<br>

- `Faster R-CNN`: 物体候補を作り、クラスと位置を予測する検出モデル
- `ResNet50 + FPN`: 画像特徴を取り出すバックボーン
- `weights=...DEFAULT`: 事前学習済みの重みを利用する

::conc::
分類の転移学習と同じように、検出でも学習済み特徴を再利用できる

---

::title::
転移学習で差し替える場所

::default::
<Transform :scale="0.86">

```python
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

num_classes = 2  # background + face

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    num_classes
)
```

</Transform>

<br>

- 検出モデルには背景クラスが必要
- 今回は `background` と `face` の2クラス
- 画像特徴を作る部分は事前学習済み重みを利用する
- 最後の予測器だけ、今回のクラス数に合わせて差し替える

::conc::
分類では `model.fc`、検出では `roi_heads.box_predictor` を差し替える

---
layout: section
---

# 検出用DataLoader

---

::title::
今回使うNotebook

::default::
<Transform :scale="0.9">

```text
workshop/notebooks/09_detection_evaluation.ipynb
```

</Transform>

<br>

- 前回Roboflowで作成した `face` データセットを使う
- Roboflowから `coco` 形式でダウンロードする
- `train/_annotations.coco.json` を読み、PyTorch用Datasetに変換する
- `fasterrcnn_resnet50_fpn` で転移学習し、`mAP@0.5` と `mAP@0.5:0.95` を確認する

::conc::
今回の資料は、このNotebookの実装順に沿って進める

---

::title::
COCO形式の構成

::default::
<Transform :scale="0.92">

```text
dataset.location/
  train/
    image_001.jpg
    image_002.jpg
    _annotations.coco.json
  valid/
    image_101.jpg
    _annotations.coco.json
```

</Transform>

<br>

- 画像ファイルとアノテーションJSONが同じsplitディレクトリに入る
- `_annotations.coco.json` に画像情報、クラス、バウンディングボックスが入る
- Notebookでは `COCODetectionDataset` がこのJSONを読み込む

::conc::
Roboflowで作った同じデータを、今回はCOCO形式としてPyTorchに渡す

---
layout: two-cols
---

::title::
DataLoaderに渡すデータ

::left::
<span class="font-bold">画像</span>

```python
image
```

- 型: `torch.Tensor`
- 形: `[3, H, W]`
- 値: 0から1の範囲

::right::
<span class="font-bold">正解ラベル</span>

```python
target = {
    "boxes": boxes,   # [N, 4], [xmin, ymin, xmax, ymax]
    "labels": labels, # [N]
    "image_id": image_id, 
    "area": area,
    "iscrowd": iscrowd,
}
```

::conc::
前回Roboflowで作成したデータを、PyTorchでは画像とtargetのペアとして扱う

---

::title::
Datasetで行う変換

::default::
<Transform :scale="0.86">

```python
x, y, width, height = ann["bbox"]

boxes.append([
    x,
    y,
    x + width,
    y + height,
])
labels.append(category_id_to_label[ann["category_id"]])
```

</Transform>

<br>

- COCO形式のbboxは `[x, y, width, height]`
- torchvisionの検出モデルは `[xmin, ymin, xmax, ymax]` を受け取る
- Roboflowのcategory idは、`background` を除いた `1` 始まりのlabelに変換する

::conc::
Datasetの役割は、Roboflowの形式をtorchvisionが期待する形式に変えること

---

::title::
collate_fnに入るbatch

::default::
<Transform :scale="0.86">

```python
# Datasetの__getitem__は1件分を返す
image, target = train_dataset[0]

# DataLoaderはbatch_size件ぶんをlistにしてcollate_fnへ渡す
batch = [
    (image_0, target_0),
    (image_1, target_1),
    (image_2, target_2),
    (image_3, target_3),
]
```

</Transform>

<br>

- `batch` は画像だけのリストではない
- 各要素は `(image, target)` のペア
- `batch_size=4` なら、4個のペアが入る

::conc::
collate_fnは、Datasetが返した複数件のペアをミニバッチ用に並べ替える

---

::title::
zipで何が起きるか

::default::
<Transform :scale="0.86">

```python
def collate_fn(batch):
    return tuple(zip(*batch))

images, targets = collate_fn(batch)
```

```python
images  = (image_0, image_1, image_2, image_3)
targets = (target_0, target_1, target_2, target_3)
```

</Transform>

<br>

- `zip(*batch)` は、ペアの0番目同士、1番目同士をまとめる
- 画像は画像だけ、正解情報は正解情報だけに分かれる
- Faster R-CNNには、この `images` と `targets` を渡す

::conc::
通常のTensor結合ではなく、画像リストとtargetリストに分けるのがポイント

---

::title::
DataLoaderの注意点

::default::
<Transform :scale="0.9">

```python
def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)
```

</Transform>

<br>

- 画像ごとに物体数が違うため、通常のミニバッチ化ではまとめられない
- `collate_fn` で `images` と `targets` のリストに分ける
- `batch_size` はColabのGPUメモリに合わせて小さめから始める

::conc::
分類よりもデータ構造が複雑になるため、DataLoaderの作り方が変わる

---
layout: section
---

# Detectionの評価項目

---
layout: two-cols
---

::title::
分類評価と検出評価の違い

::left::
<span class="font-bold">画像分類</span>

- 1枚の画像に1つの答え
- 予測クラスが合っていれば正解
- Accuracyで評価しやすい

```text
犬画像 -> 犬
猫画像 -> 猫
```

::right::
<span class="font-bold">物体検出</span>

- 1枚の画像に複数の物体
- クラスと位置の両方を見る
- 余計な検出や見逃しも評価する

```text
face + box位置
```

::conc::
検出では「クラスが合ったか」だけでは不十分

---

::title::
IoU

::default::
<div class="grid grid-cols-2 gap-8 text-xl leading-snug">
  <div>
    <div class="font-bold mb-3">IoU (Intersection over Union)</div>
    <div>予測ボックスと正解ボックスがどれだけ重なっているかを表す値</div>
    <br>
    <div class="text-2xl font-bold text-blue-500">
      IoU = 重なった面積 / 合計面積
    </div>
  </div>
  <div>
    <div class="font-bold mb-3">判定の例</div>
    <ul>
      <li>IoUが高い: 位置がよく合っている</li>
      <li>IoUが低い: 位置がずれている</li>
      <li>IoU 0.5以上を正解とみなすことが多い</li>
    </ul>
  </div>
</div>

<br>

::conc::
物体検出では、クラスだけでなくボックスの重なりも評価する

---
layout: two-cols
---

::title::
TP / FP / FN

::left::
<span class="font-bold">正しい検出</span>

- `True Positive`
- クラスが合っている
- IoUがしきい値以上

<br>

```text
faceをfaceとして検出
位置も十分に合っている
```

::right::
<span class="font-bold">誤った検出</span>

- `False Positive`
- 存在しない物体を検出
- クラスが違う
- 位置が大きくずれている

<br>

- `False Negative`
- 本当はある物体を見逃した

::conc::
検出では「余計に検出したか」と「見逃したか」を分けて考える

---
layout: two-cols
---

::title::
PrecisionとRecall

::left::
<span class="font-bold">Precision</span>

```text
Precision = TP / (TP + FP)
```

<br>

- 検出したもののうち、正しかった割合
- 余計な検出が多いと下がる
- 「間違って検出しない力」

::right::
<span class="font-bold">Recall</span>

```text
Recall = TP / (TP + FN)
```

<br>

- 正解物体のうち、見つけられた割合
- 見逃しが多いと下がる
- 「見逃さず検出する力」

::conc::
Confidenceしきい値を変えると、PrecisionとRecallのバランスが変わる

---

::title::
APとmAP

::default::
<div class="space-y-5 text-xl leading-snug">
  <div>
    <div class="font-bold">AP (Average Precision)</div>
    <div>Confidenceしきい値を変えながらPrecisionとRecallの関係を見て、1クラスごとの検出性能をまとめた値</div>
  </div>

  <div>
    <div class="font-bold">mAP (mean Average Precision)</div>
    <div>複数クラスのAPを平均した値</div>
  </div>

  <div>
    <div class="font-bold">今回のNotebook</div>
    <div>`AP@0.5` と `mAP@0.5` を確認した後、IoU 0.50から0.95までを平均して `mAP@0.5:0.95` も計算する</div>
  </div>
</div>

::conc::
`mAP@0.5:0.95` は、位置ずれに対してより厳しい評価になる

---
layout: two-cols
---

::title::
評価で確認すること

::left::
<span class="font-bold">数値で見る</span>

- `loss`
- `Precision`
- `Recall`
- `AP`
- `mAP@0.5`
- `mAP@0.5:0.95`

<br>

数値を見れば、モデル全体の傾向を比較しやすい

::right::
<span class="font-bold">画像で見る</span>

- 検出漏れはないか
- 顔以外を囲んでいないか
- ボックス位置は自然か
- Confidenceが低すぎないか

<br>

検出タスクでは、可視化による確認も重要

::conc::
評価指標と実際の検出画像を合わせて見る

---
layout: section
---

# 演習

---

::title::
演習の流れ

::default::
1. `09_detection_evaluation.ipynb` を開く

2. Roboflowから `face` データセットを `coco` 形式で取得する

3. `COCODetectionDataset` が `image` と `target` を返すことを確認する

4. `collate_fn` を使って検出用DataLoaderを作成する

5. `fasterrcnn_resnet50_fpn` の事前学習済み重みを使って転移学習する

6. IoU、Precision、Recall、AP、mAP@0.5、mAP@0.5:0.95の意味を確認する

<br>

<span class="text-xl font-bold">ColabでPyTorchによる物体検出の転移学習を実装してみましょう</span>

---
layout: section
---

# まとめ

---

::title::
まとめ

::default::
- **torchvisionによる転移学習**

  YOLOではなく、`fasterrcnn_resnet50_fpn` の事前学習済み重みを使って検出モデルを学習した

- **Roboflowデータセット**

  前回作成したデータセットを、今回はCOCO形式で取得して `COCODetectionDataset` で読み込む

- **検出モデルの差し替え**

  `roi_heads.box_predictor` を今回のクラス数に合わせて変更する

- **Detectionの評価**

  IoU、TP/FP/FN、Precision、Recall、AP、mAP@0.5、mAP@0.5:0.95を使い、位置とクラスの両方を評価する
