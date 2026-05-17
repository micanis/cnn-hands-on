---
layout: two-cols
---

::header::
演習：MaxPoolingを手動で実装

::left::

```python
def max_pool2d(img, pool_size=2):
    h, w = img.shape[:?]
    out_h = h // ?
    out_w = w // ?
    out = np.zeros((?, ?))
    
    for i in range(out_h):
        for j in range(out_w):
            region = img[
                i*?:(i+1)*?,
                j*?:(j+1)*?
            ]
            out[i, j] = np.?(region)
    return out
```

::right::

<span class="font-bold">課題</span>

NumPyでMaxPoolingを実装せよ

<span class="font-bold">ヒント</span>

- 出力サイズは入力サイズ // pool_size
- pool_size × pool_size の領域を切り出す
- `np.max()` で最大値を取得

```python
# 使用例
img = np.array([
    [1, 3, 2, 1],
    [4, 2, 3, 1],
    [2, 4, 1, 2],
    [3, 1, 2, 3]
])
out = max_pool2d(img, pool_size=2)
# → [[4, 3], [4, 3]]
```

---
layout: two-cols
---

::header::
PyTorchでのプーリング

::left::
<span class="font-bold">nn.MaxPool2d の使い方</span>

```python
import torch
import torch.nn as nn

# MaxPooling層を定義
pool = nn.MaxPool2d(
    kernel_size=2,  # 領域サイズ
    stride=2        # 移動幅
)

# 推論 (B, C, H, W) 形式
x = torch.randn(1, 1, 28, 28)
output = pool(x)  # → (1, 1, 14, 14)
```

::right::
<span class="font-bold">ポイント</span>

- 学習するパラメータがない
- strideを省略するとkernel_sizeと同じ値になる
- サイズが半分になる（kernel=stride=2の場合）

<span class="font-bold">AveragePoolingの場合</span>

```python
pool = nn.AvgPool2d(kernel_size=2, stride=2)
```

---
layout: two-cols
---

::header::
発展演習：AveragePoolingも実装

::left::

```python
def avg_pool2d(img, pool_size=2):
    h, w = img.shape[:2]
    out_h = h // pool_size
    out_w = w // pool_size
    out = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            region = img[
                i*pool_size:(i+1)*pool_size,
                j*pool_size:(j+1)*pool_size
            ]
            out[i, j] = np.?(region)
    return out
```

::right::

<span class="font-bold">課題</span>

max_pool2dを参考にAveragePoolingを実装せよ

<span class="font-bold">ヒント</span>

- `np.mean()` で平均値を取得

```python
# 使用例
img = np.array([
    [1, 3, 2, 1],
    [4, 2, 3, 1],
    [2, 4, 1, 2],
    [3, 1, 2, 3]
])
out = avg_pool2d(img, pool_size=2)
# → [[2.5, 1.75], [2.5, 2.0]]
```
