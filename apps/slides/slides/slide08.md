---
theme: tut
layout: cover
---

::title::
CNNを用いた画像認識の仕組みと実装

::date::
2026年06月10日

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
- <span class="font-bold text-blue-500">第8回 物体検出タスクの基礎及び推論</span>

::right::
- 第9回 物体検出モデルの精度評価
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
1. <span class="font-bold">ResNet</span>

    → 残差接続を使い、深いCNNでも特徴を学習しやすくした

2. <span class="font-bold">転移学習</span>

    → ImageNetで学習済みのResNet18を使い、犬猫分類の精度を高めた

3. <span class="font-bold">これまでのタスク</span>

    → 画像全体を1つのクラスに分類してきた

4. <span class="font-bold">今回学ぶこと</span>

    → 画像の中の「どこに何があるか」を扱う<span class="text-blue-500 font-bold">物体検出</span>に進む

---
layout: section
---

# 画像分類から物体検出へ

---
src: ../pages/slide08/detection_intro.md
---

---
layout: section
---

# 検出データセットのしくみ

---
src: ../pages/slide08/dataset_design.md
---

---
layout: section
---

# 物体検出モデルの代表例

---
src: ../pages/slide08/detection_models.md
---

---
layout: section
---

# Roboflowでデータセットを作る

---
src: ../pages/slide08/roboflow_dataset.md
---

---
layout: section
---

# Colabで学習・推論する

---
src: ../pages/slide08/colab_exercise.md
---

---
layout: section
---

# まとめ

---

::title::
まとめ

::default::
- **物体検出**

  画像の中の「どこに」「何が」あるかを予測するタスク

- **検出データセット**

  画像、クラス名、バウンディングボックスをセットで用意する

- **今回のクラス**

  `face` の1クラスだけを検出し、それ以外は囲まない

- **演習**

  Roboflowで作成したデータセットを使い、YOLOで学習してWebカメラ画像から顔を検出する
