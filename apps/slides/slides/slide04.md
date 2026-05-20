---
theme: tut
layout: cover
---

::title::
CNNを用いた画像認識の仕組みと実装

::date::
2026年05月25日

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
- <span class="text-blue-500" >第4回 全結合層の実装 / ネットワークの構築</span>
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
1. <span class="font-bold">プーリング層</span>

    → 特徴マップを縮小し、位置ズレに強くする処理
    
    → MaxPooling（最大値を取る）が主流

2. <span class="font-bold">活性化関数</span>

    → 非線形性を加え、深いネットワークに意味を持たせる
    
    → ReLU（負→0、正→そのまま）が主流

3. <span class="font-bold">今回学ぶこと</span>

    → CNNの最後の層：<span class="text-blue-500 font-bold">全結合層</span>と<span class="text-blue-500 font-bold">ネットワークの構築</span>

---
layout: section
---

# 全結合層とは

---
src: ../pages/slide04/fc_layer.md
---

---
layout: section
---

# 全結合層の実装

---
src: ../pages/slide04/fc_impl.md
---

---
layout: section
---

# ネットワークの構築

---
src: ../pages/slide04/network.md
---

---
layout: section
---

# まとめ

---
src: ../pages/slide04/summary.md
---
