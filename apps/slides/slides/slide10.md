---
theme: tut
layout: cover
---

::title::
CNNを用いた画像認識の仕組みと実装

::date::
2026年06月17日

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
- 第9回 物体検出モデルの精度評価
- <span class="font-bold text-blue-500">第10回 領域分割タスクの基礎及び推論</span>
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
1. <span class="font-bold">物体検出の評価</span>

    → IoU、confidence threshold、mAP を使い、検出モデルの性能を確認した

2. <span class="font-bold">torchvision の検出モデル</span>

    → `Faster R-CNN` の事前学習済み重みを使い、クラス数に合わせて最終層を差し替えた

3. <span class="font-bold">推論の流れ</span>

    → 画像を入れると、クラス名とバウンディングボックスが返ってくる

4. <span class="font-bold">今回学ぶこと</span>

    → 物体を四角で囲む検出から一歩進み、<span class="text-blue-500 font-bold">ピクセル単位で形を捉える領域分割</span>を学ぶ

---
layout: section
---

# 画像の「どこ」を塗るか

---
src: ../pages/slide10/segmentation_intro.md
---

---
layout: section
---

# マスクの考え方

---
src: ../pages/slide10/mask_design.md
---

---
layout: section
---

# 代表的な分割モデル

---
src: ../pages/slide10/segmentation_models.md
---

---
layout: section
---

# 推論の流れ

---
src: ../pages/slide10/inference_flow.md
---

---
layout: section
---

# まとめ

---
src: ../pages/slide10/summary.md
---
