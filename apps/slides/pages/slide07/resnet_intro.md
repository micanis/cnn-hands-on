---
layout: two-cols
---

::title::
なぜResNetを使うのか

::left::
<span class="font-bold">第六回のSimpleCNN</span>

- 畳み込み層を数層重ねたシンプルなCNN

- 犬猫分類の一連の流れを理解しやすい

- ただし、表現できる特徴には限界がある

<br>

```text
Conv -> ReLU -> Pool
Conv -> ReLU -> Pool
Linear -> Linear
```

::right::
<span class="font-bold">今回のResNet</span>

- より深いCNNで複雑な特徴を学習する

- 残差接続により、深いネットワークを学習しやすくする

- 自作モデルとして構造を理解する

<br>

```text
Stem
Residual Blocks
Global Average Pooling
Linear
```

::conc::
まずは「深くすると何が変わるか」を自分で実装して確かめる

---
layout: two-cols
---

::title::
深くすると何がうれしいか

::left::
<span class="font-bold">浅い層が見る特徴</span>

- エッジ
- 色の変化
- 小さな模様
- 単純な形

<br>

画像の局所的な情報を捉える

::right::
<span class="font-bold">深い層が見る特徴</span>

- 目・耳・鼻のような部品
- 毛並みや輪郭
- 犬らしさ、猫らしさ
- 背景に左右されにくい特徴

<br>

低レベル特徴を組み合わせて、より意味のある特徴を作る

::conc::
層を深くすると、単純な特徴から複雑な特徴へ発展させやすい

---
layout: two-cols
---

::title::
ただ深くすればよいのか

::left::
<span class="font-bold">深いCNNの難しさ</span>

- 学習が不安定になりやすい

- 勾配が前の層まで伝わりにくくなる

- 層を増やしたのに精度が上がらないことがある

<br>

深くするだけでは十分ではない

::right::
<span class="font-bold">ResNetの解決策</span>

- 変換した特徴だけでなく、入力も次へ渡す

- `out + identity` で残差接続を作る

- 必要な差分だけを学習しやすくする

<br>

深いネットワークでも学習しやすい

::conc::
ResNetの中心アイデアは「入力をそのまま足し戻す」こと

---
layout: two-cols
---

::title::
残差接続のイメージ

::left::
<span class="font-bold">通常のブロック</span>

```python
y = conv_block(x)
```

<br>

<div class="flex flex-col gap-1 items-center text-sm">
  <div class="px-4 py-1 bg-gray-100 border rounded">x</div>
  <span>↓</span>
  <div class="px-4 py-1 bg-blue-100 border rounded">Conv / BN / ReLU</div>
  <span>↓</span>
  <div class="px-4 py-1 bg-gray-100 border rounded">y</div>
</div>

::right::
<span class="font-bold">ResNetのブロック</span>

```python
y = conv_block(x) + shortcut(x)
```

<br>

<div class="flex flex-col gap-1 items-center text-sm">
  <div class="px-4 py-1 bg-gray-100 border rounded">x</div>
  <span>↓</span>
  <div class="px-4 py-1 bg-blue-100 border rounded">Conv / BN / ReLU / Conv / BN</div>
  <span>+</span>
  <div class="px-4 py-1 bg-green-100 border rounded">shortcut(x)</div>
  <span>↓</span>
  <div class="px-4 py-1 bg-orange-100 border rounded">ReLU</div>
</div>


---
layout: two-cols
---

::title::
ResNet18の層数

::left::
<span class="font-bold">ResNet18の数え方</span>

```text
Stem Conv: 1層
BasicBlock: 8個 x 2 Conv = 16層
Linear: 1層
```

<br>

<span class="text-xl font-bold text-blue-500">合計 18層</span>

::right::
<span class="font-bold">今回の自作ResNet</span>

```text
layer1: BasicBlock x 2
layer2: BasicBlock x 2
layer3: BasicBlock x 2
layer4: BasicBlock x 2
```

<br>

`[2, 2, 2, 2]` の4段構成

::conc::
転移学習で使う `torchvision.models.resnet18` と対応しやすい形にする
