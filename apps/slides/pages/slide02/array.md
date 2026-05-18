---
layout: default
---

::title::
なぜ配列操作を学ぶのか

::default::

<span class="font-bold text-blue-500">画像データ = 数値の配列</span>

- 画像は縦×横×色チャンネルの<span class="font-bold">多次元配列</span>として表現される
- CNNの畳み込み処理も配列操作の組み合わせ
- NumPyを使いこなすことが画像処理・機械学習の基礎となる

---
layout: two-cols
---

::title::
Pythonリストの基礎

::left::
```python
# リストの作成
numbers = [1, 2, 3, 4, 5]

# インデックスでアクセス（0始まり）
print(numbers[0])   # 1
print(numbers[-1])  # 5（末尾）

# スライス [開始:終了:ステップ]
print(numbers[1:4])   # [2, 3, 4]
print(numbers[::2])   # [1, 3, 5]
```

::right::
<span class="font-bold">リストの作成</span>

- `[]` で囲んで要素を並べる

<span class="font-bold">インデックスアクセス</span>

- 0から始まる
- 負の値で末尾から参照

<span class="font-bold">スライス</span>

- `[開始:終了:ステップ]`
- 終了位置は含まれない

---
layout: two-cols
---

::title::
演習1: Pythonリスト

::left::

```python
# 初期配列
arr = [10, 20, 30, 40, 50]

# ここから演習
ex01 = ?
ex02 = ?
ex03 = ?

print(ex01, ex02, ex03)
```

::right::

<span class="font-bold">以下の問題を解いてみよう</span>

1. リスト `[10, 20, 30, 40, 50]` を作成し、3番目の要素を出力せよ
2. 最後の2つの要素をスライスで取り出せ
3. リストを逆順にして出力せよ

<span class="text-sm text-gray-500">ヒント: ステップに `-1` を使う</span>

---
layout: two-cols
---

::title::
NumPy配列の基礎

::left::

```python
import numpy as np

# NumPy配列の作成
arr = np.array([1, 2, 3, 4, 5])

# 形状とデータ型
print(arr.shape)  # (5,)
print(arr.dtype)  # int64

# 便利な配列生成
zeros = np.zeros(5)    # [0. 0. 0. 0. 0.]
ones = np.ones(3)      # [1. 1. 1.]
arange = np.arange(0, 10, 2)  # [0 2 4 6 8]
```

::right::

<span class="font-bold">NumPy配列の特徴</span>

- `np.array()` で作成
- `shape` で形状を確認
- `dtype` でデータ型を確認

<span class="font-bold">便利な配列生成</span>

- `np.zeros()`: ゼロ埋め
- `np.ones()`: 1埋め
- `np.arange()`: 連番生成

---
layout: two-cols
---

::title::
演習2: NumPy配列の作成

::left::

```python
import numpy as np

# ここから演習
ex01 = ?
ex02_s, ex02_d = ?, ?
ex03 = ?

print(ex01, ex02_s, ex02_d, ex03)
```

::right::

<span class="font-bold">以下の問題を解いてみよう</span>

1. `np.arange` を使って 1 から 10 までの配列を作成せよ
2. 作成した配列の `shape` と `dtype` を出力せよ
3. `np.zeros` を使って要素数7のゼロ配列を作成せよ

---
layout: two-cols
---

::title::
多次元配列（2次元）

::left::

```python
import numpy as np

# 2次元配列の作成
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(matrix.shape)  # (2, 3)

# インデックスアクセス
print(matrix[0, 1])  # 2
print(matrix[1])     # [4, 5, 6]
print(matrix[:, 0])  # [1, 4]
```

::right::

<span class="font-bold">2次元配列</span>

- 行列のように縦横にデータを持つ
- `shape` は `(行数, 列数)`

<span class="font-bold">インデックスアクセス</span>

- `[行, 列]` で要素を指定
- `[行]` で行全体
- `[:, 列]` で列全体

---

::title::
配列インデックスの可視化

::default::
<ArrayIndexDemo />

---
layout: two-cols
---

::title::
演習3: 多次元配列の操作

::left::

```python
import numpy as np
# 初期配列
arr = np.array([
  [10, 20, 30],
  [40, 50, 60],
  [70, 80, 90]
])

# ここから演習
ex01 = ?
ex02 = ?
ex03 = ?
ex04 = ?

print(ex01, ex02, ex03, ex04)
```

::right::

<span class="font-bold">以下の問題を解いてみよう</span>

1. 配列の `shape` を出力せよ
2. 中央の要素（50）を取り出せ
3. 2列目（20, 50, 80）を取り出せ
4. 右下の2×2部分を取り出せ

---
layout: two-cols
---

::title::
配列の演算

::left::

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 要素ごとの演算
print(a + b)    # [5, 7, 9]
print(a * 2)    # [2, 4, 6]
print(a * b)    # [4, 10, 18]

# 集約演算
print(a.sum())   # 6
print(a.mean())  # 2.0
print(a.max())   # 3
```

::right::

<span class="font-bold">要素ごとの演算</span>

- 配列同士の `+`, `-`, `*`, `/`
- スカラー値との演算も可能
- forループ不要で高速

<span class="font-bold">集約演算</span>

- `sum()`: 合計
- `mean()`: 平均
- `max()`, `min()`: 最大/最小

---
layout: two-cols
---

::title::
演習4: 配列の演算

::left::

```python
import numpy as np
scores = np.array([85, 90, 78, 92, 88])

# ここから演習
ex01 = ?
ex02 = ?
ex03 = ?
```

::right::

<span class="font-bold">以下の問題を解いてみよう</span>

1. 全員に5点加算した配列を作成せよ
2. 平均点を計算せよ
3. 最高点と最低点を出力せよ
