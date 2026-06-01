---
theme: tut
layout: cover
---

::title::
CNNを用いた画像認識の仕組みと実装

::date::
2026年06月01日

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
- <span class="font-bold text-blue-500">第6回 犬猫分類の学習と精度評価</span>
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
1. <span class="font-bold">損失関数</span>

    → モデルの予測と正解ラベルとの誤差を数値化する関数
    
2. <span class="font-bold">最適化（パラメータ更新）</span>

    → 損失関数を最小化するようにモデルのパラメータ（重みとバイアス）を調整するプロセス

3. <span class="font-bold">学習ループ</span>

    → データセットの読み込み、モデルの順伝播、損失計算、逆伝播、パラメータ更新の一連のサイクル
    
4. <span class="font-bold">今回学ぶこと</span>

    → これまでの知識を統合し、<span class="text-blue-500 font-bold">犬猫分類の学習と精度評価</span>を行う

---
layout: section
---

# 学習のためのプログラム設計

---
src: ../pages/slide06/dog_cat_classification.md
---

---
layout: section
---

# まとめ

---

::title::
まとめ

::default::
- **Dataset / DataLoader**

  画像ファイルとラベルを扱いやすいミニバッチに変換した

- **CNNによる2クラス分類**

  犬・猫の画像特徴を畳み込み層で抽出し、全結合層で分類した

- **精度評価**

  学習損失、検証損失、検証精度を確認し、モデルの性能を評価した

- **次回予告**

  <span class="text-blue-500 font-bold">転移学習</span>を使い、より高精度な犬猫分類に挑戦する

