---
layout: two-cols
---

::title::
これまで学んだ部品

::left::
<span class="font-bold">CNNを構成する要素</span>

| 層 | 役割 | PyTorch |
|---|---|---|
| 畳み込み | 特徴抽出 | `nn.Conv2d` |
| 活性化 | 非線形性 | `nn.ReLU` |
| プーリング | 縮小 | `nn.MaxPool2d` |
| 平坦化 | 1D変換 | `nn.Flatten` |
| 全結合 | 分類 | `nn.Linear` |

::right::
<div class="flex flex-col items-center justify-center h-full">
  <div class="text-sm mb-2">CNNの全体像</div>
  <div class="flex flex-col gap-1 items-center text-xs">
    <div class="px-3  bg-gray-100 border rounded">入力 (1, 28, 28)</div>
    <span>↓</span>
    <div class="px-3  bg-blue-100 border rounded">Conv2d</div>
    <span>↓</span>
    <div class="px-3  bg-orange-100 border rounded">ReLU</div>
    <span>↓</span>
    <div class="px-3  bg-purple-100 border rounded">MaxPool2d</div>
    <span>↓</span>
    <div class="px-3  bg-yellow-100 border rounded">Flatten</div>
    <span>↓</span>
    <div class="px-3  bg-green-100 border rounded">Linear</div>
    <span>↓</span>
    <div class="px-3  bg-gray-100 border rounded">出力 (10,)</div>
  </div>
</div>

::conc::
これらを組み合わせてネットワークを構築する

---
layout: two-cols
---

::title::
nn.Sequentialで構築

::left::
<span class="font-bold">層を順番に並べる</span>

<Transform :scale="0.85">

```python
import torch.nn as nn

model = nn.Sequential(
    # 畳み込みブロック1
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    
    # 畳み込みブロック2
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    
    # 分類部分
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 10)
)
```

</Transform>

::right::
<span class="font-bold">ポイント</span>

- 上から順に処理される
- 各層の出力が次の層の入力になる
- サイズの計算に注意

<span class="font-bold">サイズの変化</span>

<Transform :scale="0.75">

```
入力:     (1, 1, 28, 28)
Conv1:    (1, 16, 28, 28)
Pool1:    (1, 16, 14, 14)
Conv2:    (1, 32, 14, 14)
Pool2:    (1, 32, 7, 7)
Flatten:  (1, 1568)
Linear:   (1, 10)
```

</Transform>

::conc::
nn.Sequentialは単純なネットワークに便利

---
layout: two-cols
---

::title::
nn.Moduleで構築

::left::
<span class="font-bold">クラスとして定義</span>

<Transform :scale="0.9">

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x
```

</Transform>

::right::
<span class="font-bold">nn.Moduleの利点</span>

- `__init__`: 層を定義
- `forward`: 処理の流れを記述
- 複雑な構造も表現可能
- スキップ接続なども実装できる

<span class="font-bold">使い方</span>

<Transform :scale="0.8">

```python
model = SimpleCNN()
x = torch.randn(1, 1, 28, 28)
out = model(x)
print(out.shape)  # (1, 10)
```

</Transform>

::conc::
nn.Moduleは柔軟で、実際のプロジェクトでよく使われる

---
layout: two-cols
---

::title::
演習：ネットワークを構築しよう

::left::

<Transform :scale="1.0">

```python
model = nn.Sequential(
    # Conv → ReLU → Pool (ブロック1)
    nn.Conv2d(?, ?, kernel_size=3, padding=1),
    nn.?(),
    nn.MaxPool2d(?),
    
    # Conv → ReLU → Pool (ブロック2)
    nn.Conv2d(?, ?, kernel_size=3, padding=1),
    nn.?(),
    nn.MaxPool2d(?),
    
    # 分類
    nn.?(),
    nn.Linear(?, ?)
)
```

</Transform>

::right::

<span class="font-bold">課題</span>

28×28のグレースケール画像を10クラスに分類するCNNを構築せよ

<span class="font-bold">ヒント</span>

- 入力: (1, 28, 28) ← チャンネル1
- 出力: 10クラス
- Conv2d: チャンネル数を増やす
- MaxPool2d(2): サイズを半分に
- Linearの入力サイズを計算する

---

::title::
推論の実行

::default::
<span class="font-bold">モデルを使って予測する</span>

<Transform :scale="0.95">

```python
import torch

# ダミー入力（バッチサイズ1の28×28画像）
x = torch.randn(1, 1, 28, 28)

# 推論モード
model.eval()

# 予測
with torch.no_grad():
    output = model(x)
    
print(output.shape)  # (1, 10)
print(output)        # 各クラスのスコア

# 最も高いスコアのクラスを取得
predicted = output.argmax(dim=1)
print(f"予測クラス: {predicted.item()}")
```

</Transform>

::conc::
現時点では重みがランダムなので、学習前は意味のある予測はできない

---

::title::
実演

::default::
<div class="h-full w-full flex items-center justify-center"><h2>Colabで見てみよう!</h2></div>
