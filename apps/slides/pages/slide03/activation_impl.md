---
layout: two-cols
---

::title::
演習：ReLUを手動で実装

::left::

```python
def relu(x):
    return np.?(0, x)
```

<span class="font-bold">もう少し詳しく書くと</span>

```python
def relu(x):
    out = np.copy(x)
    out[out ? 0] = 0
    return out
```

::right::

<span class="font-bold">課題</span>

NumPyでReLUを実装せよ

<span class="font-bold">ヒント</span>

- `np.maximum(a, b)`: 要素ごとに大きい方を返す
- 条件付き代入でも実装可能

```python
# 使用例
x = np.array([-2, -1, 0, 1, 2])
out = relu(x)
# → [0, 0, 0, 1, 2]

# 特徴マップに適用
feature_map = np.array([
    [-1, 2, -3],
    [4, -5, 6]
])
out = relu(feature_map)
# → [[0, 2, 0], [4, 0, 6]]
```

---
layout: two-cols
---

::title::
PyTorchでの活性化関数

::left::
<span class="font-bold">nn.ReLU の使い方</span>

```python
import torch
import torch.nn as nn

# ReLU層を定義
relu = nn.ReLU()

# 推論
x = torch.tensor([-2., -1., 0., 1., 2.])
output = relu(x)
# → tensor([0., 0., 0., 1., 2.])
```

<span class="font-bold">関数としても使える</span>

```python
import torch.nn.functional as F

output = F.relu(x)
```

::right::
<span class="font-bold">その他の活性化関数</span>

```python
# Leaky ReLU（負の領域も小さな勾配）
leaky_relu = nn.LeakyReLU(negative_slope=0.01)

# Sigmoid
sigmoid = nn.Sigmoid()

# Tanh
tanh = nn.Tanh()
```

<span class="font-bold">ポイント</span>

- 活性化関数にも学習パラメータはない
- 畳み込み直後に適用するのが一般的

---
layout: two-cols
---

::title::
発展演習：Conv → ReLU → Pool

::left::

```python
import torch
import torch.nn as nn

# 各層を定義
conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
relu = nn.ReLU()
pool = nn.MaxPool2d(kernel_size=2, stride=2)

# 入力（1枚、1ch、28×28）
x = torch.randn(1, 1, 28, 28)

# 順番に適用
x = conv(x)   # → (1, 1, 28, 28)
x = relu(x)   # → (1, 1, 28, 28)
x = pool(x)   # → (1, 1, 14, 14)
```

::right::

<span class="font-bold">課題</span>

Conv2d → ReLU → MaxPool2d のパイプラインを組み、各段階のサイズを確認せよ

<span class="font-bold">確認ポイント</span>

- 畳み込み後: サイズ維持（padding=1）
- ReLU後: サイズ変化なし
- プーリング後: サイズ半減

```python
print(f"Conv後: {x.shape}")
print(f"ReLU後: {x.shape}")
print(f"Pool後: {x.shape}")
```

::conc::
これがCNNの1ブロック（Conv → ReLU → Pool）の基本形
