---
theme: tut
layout: cover
---

::title::
CNNを用いた画像認識の仕組みと実装

::date::
2026年06月22日

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
- 第10回 領域分割タスクの基礎及び推論
- <span class="font-bold text-blue-500">第11回 データ拡張の重要性</span>
- <span class="font-bold text-blue-500">第12回 エラー分析手法</span>
- 第13回 総合演習（１）
- 第14回 総合演習（２）
- 第15回 総合演習（３）
- 第16回 総合演習（４）

::conc::
第11回と第12回は、1つのスライドデックでまとめて扱います

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
1. <span class="font-bold">第10回の領域分割</span>

    → 画像の各ピクセルにクラスを割り当て、物体の輪郭を捉えた

2. <span class="font-bold">マスク</span>

    → 画像と同じサイズの正解画像を使い、ピクセル単位で学習した

3. <span class="font-bold">今回の流れ</span>

    → 学習用データを増やして汎化性能を上げ、その結果を<span class="text-blue-500 font-bold">誤分類の観点</span>で確認する

---
layout: section
---

# 第11回 データ拡張の重要性

---

::title::
データ拡張とは

::default::
- 学習時だけ画像にランダムな変化を加え、見た目のバリエーションを増やす手法

- 代表例

  - `RandomHorizontalFlip`
  - `RandomRotation`
  - `ColorJitter`

- 目的

  - 学習データへの過学習を抑える
  - 少し違う見え方の画像にも強くする
  - 犬猫分類のような小規模データで特に効く

---

::title::
学習時だけ変える

::default::
<Transform :scale="0.9">

```python
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    ),
    transforms.ToTensor(),
])
```

</Transform>

<br>

- 検証・テストでは `Random*` を入れない
- 評価時は毎回同じ入力になるようにする
- 学習だけを揺らして、分類器を頑健にする

---

::title::
11回目のNotebook

::default::
<Transform :scale="0.92">

```text
workshop/notebooks/11_data_augmentation.ipynb
```

</Transform>

<br>

- 6回目の `SimpleCNN` をそのまま使う
- `train_loader` だけにデータ拡張を入れる
- 学習曲線と検証精度を確認して、保存モデルを作る

---
layout: section
---

# 第12回 エラー分析手法

---

::title::
何を見るか

::default::
- 予測の全体精度だけでは、失敗の理由が見えにくい

- エラー分析では次を見る

  - どのクラスを間違えやすいか
  - どちらのクラスに偏って予測しているか
  - 失敗画像に共通する特徴があるか

- まずは「どのパターンで落ちるか」を把握し、その後の改善につなげる

---

::title::
混同行列

::default::
<Transform :scale="0.9">

```python
confusion = torch.zeros(2, 2, dtype=torch.int64)

for images, labels in test_loader:
    outputs = model(images.to(device))
    preds = outputs.argmax(dim=1).cpu()

    for label, pred in zip(labels, preds):
        confusion[label, pred] += 1
```

</Transform>

<br>

- 行: 正解ラベル
- 列: 予測ラベル
- 右上や左下が大きいと、クラスの取り違えが起きている

---

::title::
誤分類画像を見る

::default::
- 1枚ずつの失敗例を見ると、モデルの弱点が分かる

- 典型的には次を確認する

  - 画像が暗い / ぼけている
  - 被写体が小さい
  - 背景とクラスが似ている
  - 片方のクラスにだけ偏った撮影条件がある

---

::title::
12回目のNotebook

::default::
<Transform :scale="0.92">

```text
workshop/notebooks/12_error_analysis.ipynb
```

</Transform>

<br>

- 11回目で保存したモデルを読み込む
- `test_loader` で精度と混同行列を確認する
- 誤分類画像を並べて、改善ポイントを探す

---
layout: section
---

# まとめ

---

::title::
本日のまとめ

::default::
- **データ拡張**

  学習時だけ画像を揺らして、過学習を抑えた

- **エラー分析**

  混同行列と誤分類画像から、モデルの弱点を確認した

- **Notebook**

  `11_data_augmentation.ipynb` で学習を強化し、`12_error_analysis.ipynb` で失敗を調べる

- **次の改善**

  データ拡張とエラー分析を往復しながら、モデルとデータの両方を見直す
