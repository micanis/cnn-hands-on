---
theme: tut
layout: cover
---

::title::
CNNを用いた画像認識の仕組みと実装

::date::
2026年05月18日

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
- <span class="text-blue-500" >第3回 プーリング層の実装 / 活性化関数とは</span>
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
1. <span class="font-bold">画像とは何か</span>

    → 画像は数値の多次元配列（グレースケール: 2D、カラー: 3D）
    
    → ピクセル値は0〜255の範囲で明るさを表す

2. <span class="font-bold">畳み込みとは何か</span>

    → カーネル（フィルタ）を画像上でスライドさせながら積和演算を行う処理
    
    → エッジやパターンなどの特徴を抽出できる

3. <span class="font-bold">今回学ぶこと</span>

    → 畳み込み後の処理：<span class="text-blue-500 font-bold">プーリング</span>と<span class="text-blue-500 font-bold">活性化関数</span>

---
layout: section
---

# 環境構築の更新

---
src: ../pages/slide03/env_setting.md
---

---
layout: section
---

# プーリング層とは

---
src: ../pages/slide03/pooling.md
---

---
layout: section
---

# プーリングの実装

---
src: ../pages/slide03/pooling_impl.md
---

---
layout: section
---

# 活性化関数とは

---
src: ../pages/slide03/activation.md
---

---
layout: section
---

# 活性化関数の実装

---
src: ../pages/slide03/activation_impl.md
---

---
layout: section
---

# まとめ

---
src: ../pages/slide03/summary.md
---
