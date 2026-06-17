---
layout: two-cols
---

::title::
代表的な分割モデル

::left::
<span class="font-bold">FCN / DeepLabV3</span>

- Semantic segmentation の代表例
- 画像の各ピクセルにクラスを出す
- torchvision で事前学習済みモデルを使える

<br>

```text
画像 -> 特徴抽出 -> 各ピクセルの分類
```

::right::
<span class="font-bold">Mask R-CNN</span>

- Instance segmentation の代表例
- 物体検出と同時にマスクも出す
- 物体ごとの輪郭を分けられる

<br>

```text
画像 -> 検出 -> 各物体のマスク
```

::conc::
分割モデルは、目的に応じて semantic と instance を使い分ける

---
layout: two-cols
---

::title::
torchvision を使う理由

::left::
<span class="font-bold">今回の方針</span>

- 学習済み重みをそのまま使う
- 推論の流れを理解する
- モデルの出力形式を確認する

<br>

```python
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights
)
```

::right::
<span class="font-bold">得られるもの</span>

- すぐに動く分割モデル
- 入力前処理も weights から取得できる
- 推論結果をマスクとして表示できる

<br>

事前学習済みモデルを使うと、まず「動くもの」を作ってから中身を理解しやすい

