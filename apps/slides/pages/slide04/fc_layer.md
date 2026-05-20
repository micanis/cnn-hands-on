---
layout: two-cols
---

::title::
全結合層とは

::left::
特徴マップを1次元に変換し、**分類**を行う層

<span class="font-bold">CNNの構造おさらい</span>

1. **畳み込み層**: 特徴を抽出
2. **活性化関数**: 非線形性を追加
3. **プーリング層**: サイズを縮小
4. **全結合層**: 分類を実行 ← 今回

::right::

<div class="flex flex-col items-center justify-center h-full">
  <div class="text-sm mb-2">全結合層とは</div>
  <div class="flex flex-col gap-1 items-center text-xs">
    <div class="px-3  bg-gray-100 border rounded">Conv -> ReLU -> Pool (1)</div>
    <span>↓</span>
    <div class="px-3  bg-blue-100 border rounded">Conv -> ReLU -> Pool (2)</div>
    <span>↓</span>
    <div class="px-3  bg-orange-100 border rounded">Flatten</div>
    <span>↓</span>
    <div class="px-3  bg-purple-100 border rounded">全結合層</div>
    <span>↓</span>
    <div class="px-3  bg-yellow-100 border rounded">出力 (犬/猫)</div>
  </div>
</div>

::conc::
全結合層は抽出した特徴を使って「何であるか」を判断する

---
layout: two-cols
---

::title::
Flatten（平坦化）とは

::left::
<span class="font-bold">多次元配列を1次元に変換</span>

畳み込み・プーリング後の特徴マップは3次元
```
(チャンネル, 高さ, 幅) = (C, H, W)
```

全結合層に入力するには1次元にする必要がある
```
(C × H × W,)
```

<span class="font-bold">例</span>

```python
# 特徴マップ: (16, 7, 7)
# → Flatten後: (16 × 7 × 7) = (784,)
```

::right::
<div class="flex items-center justify-center h-full">
  <FlattenChart height="400px" width="320px" />
</div>

::conc::
Flattenは学習パラメータを持たない（形状変換のみ）

---
layout: two-cols
---

::title::
全結合層の仕組み

::left::
<span class="font-bold">すべてのニューロンが全入力と接続</span>

$y = Wx + b$

- $x$: 入力ベクトル (N次元)
- $W$: 重み行列 (M × N)
- $b$: バイアスベクトル (M次元)
- $y$: 出力ベクトル (M次元)

<span class="font-bold">例: 784次元 → 10次元</span>

パラメータ数 = 784 × 10 + 10 = **7,850**

::right::
<div class="flex items-center justify-center h-full">
  <FCLayerChart :inputNodes="4" :outputNodes="2" height="220px" width="280px" />
</div>

::conc::
全結合層は重み(W)とバイアス(b)を学習する

---

::title::
なぜ全結合層が必要か

::default::
<span class="font-bold">畳み込み層との役割の違い</span>

<table class="text-sm text-left border-collapse mt-4">
  <thead>
    <tr class="bg-blue-100">
      <th class="px-4 py-2 border"></th>
      <th class="px-4 py-2 border">畳み込み層</th>
      <th class="px-4 py-2 border">全結合層</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="px-4 py-2 border font-bold">役割</td>
      <td class="px-4 py-2 border">局所的な特徴を抽出</td>
      <td class="px-4 py-2 border">特徴を統合して分類</td>
    </tr>
    <tr class="bg-gray-50">
      <td class="px-4 py-2 border font-bold">接続</td>
      <td class="px-4 py-2 border">カーネルサイズ分の局所接続</td>
      <td class="px-4 py-2 border">全入力と全出力が接続</td>
    </tr>
    <tr>
      <td class="px-4 py-2 border font-bold">位置情報</td>
      <td class="px-4 py-2 border">保持する（2D構造）</td>
      <td class="px-4 py-2 border">破棄する（1D化）</td>
    </tr>
    <tr class="bg-gray-50">
      <td class="px-4 py-2 border font-bold">パラメータ</td>
      <td class="px-4 py-2 border">少ない（カーネル共有）</td>
      <td class="px-4 py-2 border">多い（全接続）</td>
    </tr>
  </tbody>
</table>

::conc::
畳み込みで「何がどこにあるか」を見つけ、全結合で「何であるか」を決める
