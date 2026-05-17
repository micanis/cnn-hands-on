---
layout: two-cols
---

::header::
演習：畳み込みを手動で実装

::left::

```python
def conv2d(img, kernel, padding=0):
    if padding > 0:
        img = np.pad(img, padding, mode='constant')
    h, w = img.shape[:?]
    kh, kw = kernel.?
    out_h = h - kh + ?
    out_w = w - kw + ?
    out = np.?((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            region = img[?:?+?, ?:?+?]
            out[i, j] = np.sum(region * kernel)
    return out
```

::right::

<span class="font-bold">課題</span>

NumPyで畳み込み処理を実装せよ（PyTorch互換）

<span class="font-bold">ヒント</span>

- `np.pad()` でゼロパディング
- 二重ループで出力位置を走査
- スライスで入力の一部を切り出す

```python
# 使用例
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])
out = conv2d(img, kernel, padding=1)
# padding=1, kernel=3x3 → 入出力サイズ同じ
```

---
layout: two-cols
---

::header::
PyTorchでの畳み込み

::left::
<span class="font-bold">nn.Conv2d の使い方</span>

```python
import torch
import torch.nn as nn

# 畳み込み層を定義（1ch入力で比較）
conv = nn.Conv2d(
    in_channels=1,   # 入力チャンネル
    out_channels=1,  # 出力チャンネル
    kernel_size=3,   # カーネルサイズ
    stride=1,        # ストライド
    padding=1        # パディング
)

# 推論 (B, C, H, W) 形式
x = torch.randn(1, 1, 28, 28)
output = conv(x)  # → (1, 1, 28, 28)
```

::right::
<span class="font-bold">パラメータの意味</span>

| パラメータ | 説明 |
|-----------|------|
| in_channels | 入力のチャンネル数 |
| out_channels | カーネルの数（出力ch） |
| kernel_size | カーネルのサイズ |
| stride | 移動幅 |
| padding | ゼロ埋めの幅 |

<span class="font-bold">ポイント</span>

- カーネルの値は自動で初期化（学習で最適化）
- `padding=1, kernel=3` で入出力サイズ同じ
- 出力ch = 異なる特徴を抽出するフィルタ数


