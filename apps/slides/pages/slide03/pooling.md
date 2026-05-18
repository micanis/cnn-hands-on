---
layout: two-cols
---

::title::
プーリングとは

::left::
特徴マップを縮小し、位置のズレに強くする処理

<span class="font-bold">なぜ必要か</span>

1. **計算量の削減**: サイズを小さくして効率化
2. **位置不変性**: 特徴の位置が少しズレても検出可能
3. **過学習の防止**: パラメータ数を減らす

::right::
<div class="flex items-center justify-center h-full">
  <PoolingDemo type="max" />
</div>

::conc::
2×2領域ごとに最大値を取る → サイズが半分に

---
layout: two-cols
---

::title::
MaxPooling vs AveragePooling

::left::
<div class="flex flex-col items-center">
  <span class="font-bold mb-4">MaxPooling（最大値）</span>
  <PoolingDemo type="max" />
  <div class="mt-4 text-sm text-gray-600">最も顕著な特徴を保持</div>
</div>

::right::
<div class="flex flex-col items-center">
  <span class="font-bold mb-4">AveragePooling（平均値）</span>
  <PoolingDemo type="avg" />
  <div class="mt-4 text-sm text-gray-600">滑らかな特徴を保持</div>
</div>

::conc::
基本はMaxPoolingを使用する（エッジなど重要な特徴を残すため）

---

::title::
プーリングのパラメータ

::default::
<span class="font-bold">典型的な設定</span>

| パラメータ | 値 | 意味 |
|-----------|---|------|
| kernel_size | 2 | 2×2の領域をまとめる |
| stride | 2 | 2pxずつ移動（重なりなし） |

<br>

<span class="font-bold">出力サイズの計算</span>

$$
\text{出力サイズ} = \frac{\text{入力サイズ}}{\text{stride}} = \frac{H}{2}
$$

例: 入力 28×28、kernel=2、stride=2 → 出力 14×14

::conc::
プーリングには学習するパラメータがない（単純な演算のみ）

---

::title::
実演

::default::
<div class="h-full w-full flex items-center justify-center"><h2>Colabで見てみよう!</h2></div>
