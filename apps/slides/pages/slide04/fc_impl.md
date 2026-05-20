---
layout: two-cols
---

::title::
演習：Flattenを手動で実装

::left::
<Transform :scale="0.9">

```python
def flatten(x):
    """
    3次元配列を1次元に変換
    入力: (C, H, W)
    出力: (C * H * W,)
    """
    return x.?(?)
```

</Transform>

<span class="font-bold">使用例</span>

```python
x = np.random.randn(16, 7, 7)
flat = flatten(x)
print(flat.shape)  # (784,)
```

::right::

<span class="font-bold">課題</span>

NumPyでFlattenを実装せよ

<span class="font-bold">ヒント</span>

- `np.reshape()` または `.reshape()` を使用
- `-1` を指定すると自動計算される

```python
# reshapeの例
arr = np.array([[1, 2], [3, 4]])
arr.reshape(-1)  # → [1, 2, 3, 4]
```

---
layout: two-cols
---

::title::
演習：全結合層を手動で実装

::left::
<Transform :scale="0.9">

```python
def linear(x, W, b):
    """
    全結合層の順伝播
    x: 入力 (N,)
    W: 重み (M, N)
    b: バイアス (M,)
    出力: (M,)
    """
    return np.?(?, ?) + ?
```

</Transform>

<span class="font-bold">使用例</span>

```python
x = np.random.randn(784)
W = np.random.randn(10, 784)
b = np.random.randn(10)
out = linear(x, W, b)  # (10,)
```

::right::

<span class="font-bold">課題</span>

NumPyで全結合層を実装せよ

<span class="font-bold">ヒント</span>

- 行列とベクトルの積には `np.dot()` を使用
- $y = Wx + b$ を実装する

```python
# np.dotの例
W = np.array([[1, 2], [3, 4]])
x = np.array([1, 1])
np.dot(W, x)  # → [3, 7]
```

---
layout: two-cols
---

::title::
PyTorchでの全結合層

::left::
<span class="font-bold">nn.Flatten & nn.Linear の使い方</span>

<Transform :scale="1.0">

```python
import torch
import torch.nn as nn

flatten = nn.Flatten()

# (B, C, H, W) → (B, C*H*W)
x = torch.randn(1, 16, 7, 7)
flat = flatten(x)
print(flat.shape)  # (1, 784)


fc = nn.Linear(784, 10)
out = fc(flat)
print(out.shape)  # (1, 10)
```

</Transform>


::right::
<span class="font-bold">ポイント</span>

- `nn.Flatten()` はバッチ次元を保持する
- `nn.Linear(in_features, out_features)`
  - in_features: 入力の次元数
  - out_features: 出力の次元数
- 重みとバイアスは自動で初期化される

```python
# パラメータの確認
print(fc.weight.shape)  # (10, 784)
print(fc.bias.shape)    # (10,)
```

::conc::
PyTorchでは重みの初期化や形状管理を自動で行ってくれる

---
layout: two-cols
---

::title::
発展演習：複数の全結合層

::left::
<span class="font-bold">隠れ層を追加する</span>

```python
# 784 → 128 → 10
fc1 = nn.Linear(784, 128)
relu = nn.ReLU()
fc2 = nn.Linear(128, 10)

x = torch.randn(1, 784)
h = relu(fc1(x))  # 隠れ層
out = fc2(h)      # 出力層
print(out.shape)  # (1, 10)
```

::right::
<span class="font-bold">なぜ隠れ層を入れるか</span>

- より複雑なパターンを学習できる
- 全結合層も活性化関数で非線形性を追加
- 層が増える = パラメータ増 = 表現力向上

<div class="flex flex-col items-center mt-4">
  <div class="flex items-center gap-2 text-xs">
    <div class="px-2 py-1 bg-blue-200 rounded">784</div>
    <span>→</span>
    <div class="px-2 py-1 bg-orange-200 rounded">128</div>
    <span>→ReLU→</span>
    <div class="px-2 py-1 bg-green-200 rounded">10</div>
  </div>
</div>

::conc::
全結合層の間にも活性化関数を挟むのが一般的
