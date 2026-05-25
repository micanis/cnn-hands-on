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
- 第4回 全結合層の実装 / ネットワークの構築
- <span class="text-blue-500" >第5回 損失関数と最適化 / MNISTで学習と推論</span>
- 第6回 犬猫分類の学習と精度評価
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
1. <span class="font-bold">全結合層</span>

    → 特徴マップを1次元化(Flatten)し、分類を行う層
    
    → $y = Wx + b$ の行列演算、すべての入力と出力が接続

2. <span class="font-bold">ネットワークの構築</span>

    → Conv → ReLU → Pool → ... → Flatten → Linear の流れ
    
    → `nn.Sequential` で簡単に、`nn.Module` で柔軟に構築

3. <span class="font-bold">今回学ぶこと</span>

    → <span class="text-blue-500 font-bold">損失関数</span>と<span class="text-blue-500 font-bold">最適化</span>を学び、<span class="text-blue-500 font-bold">MNISTで実際に学習</span>する

---
layout: section
---

# 損失関数とは

---
src: ../pages/slide05/loss_function.md
---

---
layout: section
---

# 最適化（パラメータ更新）

---
src: ../pages/slide05/optimization.md
---

---
layout: section
---

# MNISTデータセット

---
src: ../pages/slide05/mnist.md
---

---
layout: section
---

# 学習ループの実装

---
src: ../pages/slide05/training_loop.md
---

---
layout: section
---

# まとめ

---
src: ../pages/slide05/summary.md
---
