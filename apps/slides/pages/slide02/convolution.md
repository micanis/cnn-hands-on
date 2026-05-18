---
layout: two-cols
---

::title::
畳み込みとは

::left::
画像の上を小さな窓（カーネル）がスライドしながら、局所的な特徴を抽出する処理

<span class="font-bold">処理の流れ</span>

1. カーネルを画像の左上に配置
2. 重なった部分の要素同士を掛け算
3. 全ての積を足し合わせる
4. カーネルを1ピクセル移動して繰り返す

::right::
<div class="flex items-center justify-center gap-3 h-full">
  <div>
    <div class="text-xs text-center mb-1">入力画像</div>
    <div class="grid grid-cols-5 gap-px bg-gray-300 p-px text-xs">
      <div class="w-5 h-5 bg-blue-100 border border-blue-400 flex items-center justify-center">1</div>
      <div class="w-5 h-5 bg-blue-100 border border-blue-400 flex items-center justify-center">2</div>
      <div class="w-5 h-5 bg-blue-100 border border-blue-400 flex items-center justify-center">3</div>
      <div class="w-5 h-5 bg-white flex items-center justify-center">4</div>
      <div class="w-5 h-5 bg-white flex items-center justify-center">5</div>
      <div class="w-5 h-5 bg-blue-100 border border-blue-400 flex items-center justify-center">2</div>
      <div class="w-5 h-5 bg-blue-100 border border-blue-400 flex items-center justify-center">3</div>
      <div class="w-5 h-5 bg-blue-100 border border-blue-400 flex items-center justify-center">4</div>
      <div class="w-5 h-5 bg-white flex items-center justify-center">5</div>
      <div class="w-5 h-5 bg-white flex items-center justify-center">6</div>
      <div class="w-5 h-5 bg-blue-100 border border-blue-400 flex items-center justify-center">3</div>
      <div class="w-5 h-5 bg-blue-100 border border-blue-400 flex items-center justify-center">4</div>
      <div class="w-5 h-5 bg-blue-100 border border-blue-400 flex items-center justify-center">5</div>
      <div class="w-5 h-5 bg-white flex items-center justify-center">6</div>
      <div class="w-5 h-5 bg-white flex items-center justify-center">7</div>
      <div class="w-5 h-5 bg-white flex items-center justify-center">4</div>
      <div class="w-5 h-5 bg-white flex items-center justify-center">5</div>
      <div class="w-5 h-5 bg-white flex items-center justify-center">6</div>
      <div class="w-5 h-5 bg-white flex items-center justify-center">7</div>
      <div class="w-5 h-5 bg-white flex items-center justify-center">8</div>
      <div class="w-5 h-5 bg-white flex items-center justify-center">5</div>
      <div class="w-5 h-5 bg-white flex items-center justify-center">6</div>
      <div class="w-5 h-5 bg-white flex items-center justify-center">7</div>
      <div class="w-5 h-5 bg-white flex items-center justify-center">8</div>
      <div class="w-5 h-5 bg-white flex items-center justify-center">9</div>
    </div>
  </div>
  <div class="text-xl">*</div>
  <div>
    <div class="text-xs text-center mb-1">カーネル</div>
    <div class="grid grid-cols-3 gap-px bg-blue-400 p-px text-xs">
      <div class="w-5 h-5 bg-blue-100 flex items-center justify-center font-bold">1</div>
      <div class="w-5 h-5 bg-blue-100 flex items-center justify-center font-bold">0</div>
      <div class="w-5 h-5 bg-blue-100 flex items-center justify-center font-bold">-1</div>
      <div class="w-5 h-5 bg-blue-100 flex items-center justify-center font-bold">1</div>
      <div class="w-5 h-5 bg-blue-100 flex items-center justify-center font-bold">0</div>
      <div class="w-5 h-5 bg-blue-100 flex items-center justify-center font-bold">-1</div>
      <div class="w-5 h-5 bg-blue-100 flex items-center justify-center font-bold">1</div>
      <div class="w-5 h-5 bg-blue-100 flex items-center justify-center font-bold">0</div>
      <div class="w-5 h-5 bg-blue-100 flex items-center justify-center font-bold">-1</div>
    </div>
  </div>
  <div class="text-xl">=</div>
  <div>
    <div class="text-xs text-center mb-1">出力</div>
    <div class="grid grid-cols-3 gap-px bg-gray-300 p-px text-xs">
      <div class="w-5 h-5 bg-green-100 flex items-center justify-center font-bold">-6</div>
      <div class="w-5 h-5 bg-white flex items-center justify-center"></div>
      <div class="w-5 h-5 bg-white flex items-center justify-center"></div>
      <div class="w-5 h-5 bg-white flex items-center justify-center"></div>
      <div class="w-5 h-5 bg-white flex items-center justify-center"></div>
      <div class="w-5 h-5 bg-white flex items-center justify-center"></div>
      <div class="w-5 h-5 bg-white flex items-center justify-center"></div>
      <div class="w-5 h-5 bg-white flex items-center justify-center"></div>
      <div class="w-5 h-5 bg-white flex items-center justify-center"></div>
    </div>
  </div>
</div>


---

::title::
カーネル（フィルタ）

::default::
カーネルはどんな特徴を検出するかを決める小さな行列

<span class="font-bold">エッジ検出カーネルの例</span>

```
横エッジ:        縦エッジ:
[-1, -1, -1]    [-1, 0, 1]
[ 0,  0,  0]    [-1, 0, 1]
[ 1,  1,  1]    [-1, 0, 1]
```

<span class="font-bold">ぼかしカーネルの例</span>

```
[1/9, 1/9, 1/9]
[1/9, 1/9, 1/9]
[1/9, 1/9, 1/9]
```

CNNではカーネルの値を学習によって自動で獲得する

---
layout: two-rows
---

::title::
畳み込みの数式

::top::
<span class="font-bold">畳み込みの定義式</span>

$$
(I * K)(i, j) = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I(i+m, j+n) \cdot K(m, n)
$$

$I$: 入力画像、$K$: カーネル（$M \times N$）、$(i, j)$: 出力位置、$(m, n)$: カーネル内位置

::bottom::
<span class="font-bold">計算例</span>（入力の左上3×3、縦エッジ検出カーネル）

$$
1 \cdot 1 + 2 \cdot 0 + 3 \cdot (-1) + 2 \cdot 1 + 3 \cdot 0 + 4 \cdot (-1) + 3 \cdot 1 + 4 \cdot 0 + 5 \cdot (-1) = -6
$$

::conc::
要素ごとの積和を画像全体で繰り返す

---
layout: two-cols
---

::title::
ストライドとパディング

::left::
<span class="font-bold">ストライド（stride）</span>

カーネルを移動させる幅

- stride=1: 1pxずつ移動
- stride=2: 2pxずつ（出力が半分）

<span class="font-bold">パディング（padding）</span>

入力の周囲を0で埋める

- padding=0: 出力が小さくなる
- padding=1: 出力サイズを維持

::right::

<span class="font-bold">例: 入力7×7、カーネル3×3</span>

| 設定 | 出力 |
|------|------|
| s=1, p=0 | 5×5 |
| s=1, p=1 | 7×7 |
| s=2, p=0 | 3×3 |

---

::title::
畳み込みの実演

::default::
<ConvolutionDemo />
