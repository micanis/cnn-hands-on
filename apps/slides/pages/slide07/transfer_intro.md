---
layout: two-cols
---

::title::
自作ResNetの次に考えること

::left::
<span class="font-bold">自作ResNetで改善できること</span>

- SimpleCNNより深い特徴を学習できる

- 残差接続により学習しやすい

- 犬猫分類で85-90%前後を狙える

::right::
<span class="font-bold">まだ残る課題</span>

- 重みはランダム初期化から学習する

- 犬猫データだけでは学習できる知識に限界がある

- 高精度には多くのデータと時間が必要

::conc::
構造を良くしても、学習に使えるデータ量には限界がある

---

::title::
転移学習とは

::default::
**転移学習 (Transfer Learning)**

- 大量のデータで学習済みのモデルを、別のタスクに再利用する方法

- 今回は ImageNet で学習済みのResNet18を、犬猫分類に使う

- 画像の基本的な特徴抽出は再利用し、最後の分類部分だけを犬猫用に変える

<br>

**なぜ有効か**

- 犬猫データが少なくても、すでに学習済みの画像特徴を使える

- 自作ResNetを一から学習するより、少ないデータで高精度になりやすい

- 今回は98-100%前後を目標にする

---
layout: two-cols
---

::title::
ImageNetで学習済みの特徴

::left::
<span class="font-bold">ImageNet</span>

- 多数の一般物体画像を含む大規模データセット

- 犬や猫だけでなく、車、鳥、道具、食べ物なども含む

- ResNet18はこのデータから画像の特徴を学習済み

::right::
<span class="font-bold">再利用できる特徴</span>

- エッジ
- 色や模様
- 目・耳・輪郭のような部品
- 物体らしい形

<br>

犬猫分類でも、これらの特徴は役に立つ

::conc::
「犬猫専用の知識」ではなく「画像を見るための一般的な知識」を借りる

---
layout: two-cols
---

::title::
転移学習で変える場所

::left::
<span class="font-bold">事前学習済みResNet18</span>

```text
Input image
  ↓
Feature extractor
  ↓
fc: 1000 classes
```

ImageNetの1000クラス分類用

::right::
<span class="font-bold">犬猫分類用に変更</span>

```text
Input image
  ↓
Feature extractor
  ↓
fc: 2 classes
```

特徴抽出器は再利用し、最後の `fc` だけ差し替える

::conc::
転移学習では「特徴抽出器を借りて、分類器だけ作り直す」
