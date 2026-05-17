---
layout: two-cols
---

::header::
活性化関数とは

::left::
ニューロンの出力に**非線形性**を加える関数

<span class="font-bold">なぜ必要か</span>

畳み込みは線形演算の積み重ね

$$
y = W_2(W_1 x) = (W_2 W_1) x = W' x
$$

何層重ねても1層と同じ表現力しかない

**活性化関数が非線形性を加えることで、深いネットワークが意味を持つ**

::right::
<div class="flex flex-col items-center justify-center h-full gap-4">
  <div class="text-center">
    <div class="text-sm mb-2">活性化関数なし</div>
    <div class="flex items-center gap-2">
      <div class="w-12 h-8 bg-blue-100 border rounded flex items-center justify-center text-xs">線形</div>
      <span>→</span>
      <div class="w-12 h-8 bg-blue-100 border rounded flex items-center justify-center text-xs">線形</div>
      <span>=</span>
      <div class="w-12 h-8 bg-blue-200 border rounded flex items-center justify-center text-xs">線形</div>
    </div>
  </div>
  <div class="text-center">
    <div class="text-sm mb-2">活性化関数あり</div>
    <div class="flex items-center gap-2">
      <div class="w-12 h-8 bg-blue-100 border rounded flex items-center justify-center text-xs">線形</div>
      <span>→</span>
      <div class="w-12 h-8 bg-orange-200 border rounded flex items-center justify-center text-xs">非線形</div>
      <span>→</span>
      <div class="w-12 h-8 bg-green-200 border rounded flex items-center justify-center text-xs">複雑</div>
    </div>
  </div>
</div>

::conc::
活性化関数によって、CNNは複雑なパターンを学習できる

---
layout: three-cols
---

::header::
代表的な活性化関数

::left::
<div class="flex flex-col items-center h-full">
  <div class="text-lg font-bold mb-2">Sigmoid</div>
  <ActivationChart type="sigmoid" color="#f97316" height="180px" />
  <div class="text-sm mt-2">出力: 0〜1</div>
  <div class="text-xs text-gray-500 mt-1">勾配消失しやすい</div>
</div>

::center::
<div class="flex flex-col items-center h-full">
  <div class="text-lg font-bold mb-2">Tanh</div>
  <ActivationChart type="tanh" color="#a855f7" height="180px" />
  <div class="text-sm mt-2">出力: -1〜1</div>
  <div class="text-xs text-gray-500 mt-1">Sigmoidより中心化</div>
</div>

::right::
<div class="flex flex-col items-center h-full">
  <div class="text-lg font-bold mb-2 text-blue-600">ReLU（主流）</div>
  <ActivationChart type="relu" color="#3b82f6" height="180px" />
  <div class="text-sm mt-2">負→0、正→そのまま</div>
  <div class="text-xs text-gray-500 mt-1">計算単純、勾配消失に強い</div>
</div>

::conc::
現代のCNNではほぼReLUを使用する

---
layout: two-cols
---

::header::
ReLUが主流な理由

::left::
<span class="font-bold">ReLU (Rectified Linear Unit)</span>

$$
\text{ReLU}(x) = \max(0, x) = \begin{cases} x & (x > 0) \\ 0 & (x \leq 0) \end{cases}
$$

<span class="font-bold">メリット</span>

1. **計算が単純**: 比較と代入だけ
2. **勾配消失に強い**: 正の領域で勾配が1
3. **スパース性**: 負の値を0にして疎な表現

::right::
<div class="flex flex-col items-center justify-center h-full">
  <div class="text-sm font-bold mb-2 text-center">ReLUのグラフ</div>
  <ActivationChart type="relu" color="#3b82f6" height="220px" width="100%" />
</div>

::conc::
負の値を0に、正の値はそのまま通す

---

::header::
実演

::default::
<div class="h-full w-full flex items-center justify-center"><h2>Colabで見てみよう!</h2></div>
