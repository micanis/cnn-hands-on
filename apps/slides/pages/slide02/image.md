---
layout: two-cols
---

::title::
画像は数値である

::left::
コンピュータにとって、画像は<span class="text-blue-500 font-bold">数値の集まり</span>

<span class="font-bold">人間の目には</span>
- 色や形として認識
- 「犬」「猫」などの意味を持つ

<span class="font-bold">コンピュータにとっては</span>
- 0〜255の数値の配列
- 数値の並びに意味はない

::right::
<div class="flex items-center justify-center h-full">
<img src="/slide02/pixel-grid.png" alt="ピクセルと数値の対応" class="h-48" />
</div>

::conc::
画像の最小単位を<span class="font-bold">ピクセル</span>と呼び、各ピクセルが数値を持つ

---

::title::
RGBチャンネルの可視化

::default::
<ImageChannelDemo />

::conc::
カラー画像は R・G・B 各0〜255の3チャンネルで構成される

---
layout: two-cols
---

::title::
PyTorchライブラリでの型

::left::
PyTorchでは画像を<span class="text-blue-500 font-bold">Tensor（テンソル）</span>として扱う

```python
import torch
from torchvision import transforms
from PIL import Image

# 画像を読み込んでTensorに変換
image = Image.open("cat.jpg")
tensor = transforms.ToTensor()(image)

print(tensor.shape)
# torch.Size([3, 224, 224])
# チャンネル数, 高さ, 幅
```

::right::
| 項目 | 説明 |
|------|------|
| 形状 | <span class="text-sm">(C, H, W) = (チャンネル, 高さ, 幅)</span> |
| 値の範囲 | 0.0 〜 1.0 |
| 型 | torch.float32 |


<span class="font-bold">ToTensor()の役割</span>

- PIL画像 → Tensor に変換
- 0〜255 → 0.0〜1.0 に正規化
- (H, W, C) → (C, H, W) に並び替え

---
layout: two-cols
---

::title::
演習の準備：画像を配列として読み込む

::left::
<span class="font-bold">NumPyで画像を読み込む</span>

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 画像を読み込んでNumPy配列に変換
img = Image.open("cat.jpg")
arr = np.array(img)

print(arr.shape)  # (高さ, 幅, 3)
print(arr.dtype)  # uint8 (0-255)
```

::right::
<span class="font-bold">配列の構造</span>

- <code>arr[y, x, c]</code> でアクセス
- y: 縦位置（0が上）
- x: 横位置（0が左）
- c: チャンネル（0=R, 1=G, 2=B）

```python
# 例: (100, 50) の赤成分
arr[100, 50, 0]
```

::conc::
画像 = 3次元配列として操作できる

---
layout: two-cols
---

::title::
演習1：グレースケール変換（手動）

::left::

```python
# RGB平均でグレースケール化
gray = arr.?(axis=?)

print(gray.shape)  # (高さ, 幅)

# 表示
plt.imshow(gray, cmap='gray')
plt.show()
```

::right::

<span class="font-bold">課題</span>

RGB値の平均を取ってグレースケール画像を作成せよ

<span class="font-bold">ヒント: axis=? の意味</span>

- axis=0: 縦方向に平均
- axis=1: 横方向に平均
- axis=2: チャンネル方向に平均

---
layout: two-cols
---

::title::
演習2：画像の反転（手動）

::left::

```python
# 左右反転
flipped_lr = arr[?, ?, ?]

plt.imshow(flipped_lr)
plt.show()
```

::right::

<span class="font-bold">課題</span>

スライスを使って画像を左右反転せよ

<span class="font-bold">ヒント: 逆順スライス</span>

<code>::-1</code> は逆順を意味する

```python
a = [1, 2, 3, 4, 5]
a[::-1]  # [5, 4, 3, 2, 1]
```

---
layout: two-cols
---

::title::
演習3：明るさ調整（手動）

::left::

```python
# 明るくする（+50）
bright = ?

# 値が255を超える場合の処理
bright = np.clip(bright, 0, 255)
bright = bright.astype(np.?)

plt.imshow(bright)
plt.show()
```

::right::

<span class="font-bold">課題</span>

全ピクセルに50を加算して画像を明るくせよ

<span class="font-bold">ヒント</span>

- ブロードキャスト: <code>arr + 50</code>
- ブロードキャスト時のオーバーフローに注意すること
- 値が255を超えないよう <code>np.clip</code> を使う
- 型を <code>uint8</code> に戻す

::conc::
配列操作だけで基本的な画像処理ができる


---

::title::
暇な人向け

::default::
```python
word = "
    JNPGCZBUXHJAVWXGWIZAXTIQYMRRSSYDNUWCJYVZVZZCYZYKWUMOJNZYUJIKCWXUVDDNOYJDXYIXADXJYZNZTSNQDXGUBYSZPRCRPQYIPTXCSIHNZXWFWSQKVYOHWIZJYWZDQSLPIFXRYWYLXWWYDCBWIKJQGWSUXPHCORZXSXLWWOIZPIMQXCWVCMAYWKKPRNWAYYATXCHQCZKTIWIRLOZVQWKXZGYRZUQJXDJQQYMYLNBZXWWMJXPZXKYPGWRETBPPDHUMQMKNUYHFGQKHMYKJKWYTIBZSTOZFHLQVYXLGCNIEXQFAGBWAFMXSWXTCWZKXSAXUZFLUYPWIGKWYUDTOOYYWZYQZXDVJSYSTGJWXNZGZOZSZCXCHZERWCIWYTIPQRWXZWCYYQYUWTNGZXZUBYKYVZWPEKOYZNWKYGPOYXLTWYYTAFYXPXXQWCWSZLMXRGKVCCWLANWWCBZYWLIRYGJRHMKWVBWXWGRLETQNZHYAQUTZK
"

# 以下の操作をwordに対して行ってください
# 1. 全てのWを削除する
# 2. Xという文字の3つ先がX(これ)でないなら、XをEに変えてください
# 3. 全てのEYという文字をXに変えてください
```

---
layout: two-cols
---

::title::
ライブラリを使うとどうなる？

::left::
<span class="font-bold">torchvision.transforms</span>

```python
from torchvision import transforms

# グレースケール変換
gray = transforms.Grayscale()(img)

# 左右反転
flip = transforms.RandomHorizontalFlip(p=1)(img)

# 明るさ調整
bright = transforms.ColorJitter(
    brightness=0.5
)(img)
```

<span class="font-bold">同じ処理を頭を使わずに書ける</span>

::right::
<span class="font-bold">ライブラリの利点</span>

1. コードが短い（数十行→1行）
2. 高速に動作（最適化済み）
3. エッジケース対応済み
4. Composeで簡単に組み合わせ

::conc::
仕組みを理解した上でライブラリを使おう
