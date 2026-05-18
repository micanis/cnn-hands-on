---
theme: tut
layout: cover
---

::title::
CNNを用いた画像認識の仕組みと実装

::date::
2026年05月11日

::default::
山中春輝

---
layout: two-cols
---

::title::
授業計画

::left::
- 第1回 ガイダンス / 犬猫の分類を試してみよう
- <span class="text-blue-500" >第2回 画像とは / 畳み込みの実装</span>
- 第3回 活性化関数とは / プーリング層の実装
- 第4回 全結合層の実装 / CNNのモデル構築
- 第5回 損失関数とは / パラメータ更新の基礎
- 第6回 モデルの学習ループ実装と精度評価
- 第7回 転移学習の実装と精度評価
- 第8回 物体検出タスクの基礎及び推論

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
1. <span class="font-bold">CNNとはなにか</span>

    → 人間の視覚野をモデルにした、画像認識に特化したニューラルネットワーク
    
    → 畳み込み層・プーリング層・全結合層の3つで構成される

2. <span class="font-bold">画像分類とはなにか</span>

    → 入力画像が、あらかじめ定義されたどのクラス (ラベル) に属するかを予測するタスク
    
    → 前回は学習済みモデルを使って犬猫分類の推論を体験した

3. <span class="font-bold">今回学ぶこと</span>

    → CNNの最初のステップである<span class="text-blue-500 font-bold">畳み込み</span>の仕組みと実装

---
layout: section
---

# 配列操作を学ぶ

---
src: ../pages/slide02/array.md
---

---
layout: section
---

# 画像とは

---
src: ../pages/slide02/image.md
---

---
layout: section
---

# 畳み込み(Convolution)とは

---
src: ../pages/slide02/convolution.md
---

---
layout: section
---

# 畳み込みの実装

---
src: ../pages/slide02/implementation.md
---

---
layout: section
---

# まとめ

---
src: ../pages/slide02/summary.md
---
