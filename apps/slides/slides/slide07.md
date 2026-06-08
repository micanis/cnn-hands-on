---
theme: tut
layout: cover
---

::title::
CNNを用いた画像認識の仕組みと実装

::date::
2026年06月08日

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
- <span class="font-bold text-blue-500">第7回 ResNetと転移学習による精度改善</span>
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
1. <span class="font-bold">犬猫分類データセット</span>

    → `CDDataset` と `DataLoader` で画像とラベルをミニバッチ化した

2. <span class="font-bold">SimpleCNN</span>

    → 畳み込み層で特徴を抽出し、全結合層で犬・猫の2クラスに分類した

3. <span class="font-bold">学習と評価</span>

    → Train Loss、Validation Loss、Validation Accuracyで性能を確認した

4. <span class="font-bold">今回学ぶこと</span>

    → 自作ResNetでネットワークを深くし、その後<span class="text-blue-500 font-bold">転移学習</span>でさらに精度を上げる

---
layout: section
---

# ResNetとは

---
src: ../pages/slide07/resnet_intro.md
---

---
layout: section
---

# ResNetの実装

---
src: ../pages/slide07/resnet_impl.md
---

---
layout: section
---

# 転移学習とは

---
src: ../pages/slide07/transfer_intro.md
---

---
layout: section
---

# 転移学習の実装

---
src: ../pages/slide07/transfer_impl.md
---

---
layout: section
---

# まとめ

---
layout: two-cols
---

::title::
まとめ

::left::
<div class="space-y-4 text-xl leading-snug">
  <div>
    <div class="font-bold">自作ResNet18</div>
    <div>残差接続を使い、SimpleCNNより深いネットワークで犬猫分類を行った</div>
  </div>

  <div>
    <div class="font-bold">残差接続</div>
    <div><code>out + identity</code> によって、深いモデルでも特徴と勾配を伝えやすくした</div>
  </div>

  <div>
    <div class="font-bold">精度比較</div>
    <div>自作ResNetは85-90%前後、転移学習は98-100%前後を目標に比較する</div>
  </div>
</div>

::right::
<div class="space-y-4 text-xl leading-snug">
  <div>
    <div class="font-bold">転移学習</div>
    <div>ImageNetで学習済みのResNet18を使い、画像特徴を再利用した</div>
  </div>

  <div>
    <div class="font-bold">最終層の差し替え</div>
    <div><code>model.fc</code> を犬猫2クラス分類用に置き換え、分類器だけを学習した</div>
  </div>

  <div>
    <div class="font-bold">次回予告</div>
    <div>物体検出に進み、画像の中の「どこに何があるか」を扱う</div>
  </div>
</div>
