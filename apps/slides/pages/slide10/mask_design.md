---
layout: two-cols
---

::title::
マスクとは

::left::
<span class="font-bold">マスク</span>

- 各ピクセルに付けるラベル画像
- 画像と同じ高さ・幅を持つ
- 色や番号でクラスを表す

<br>

```text
0 0 0 0 1 1 1
0 0 0 0 1 1 1
0 2 2 2 2 1 1
```

<br>

ここでは `0=background`、`1=person`、`2=road` のように読む

::right::
<div class="h-full flex items-center justify-center">
<img src="/public/slide10/mask.jpeg" class="block mx-auto" />
</div>

::conc::
領域分割では、画像そのものとは別に「正解の塗り分け」を用意する

---
layout: two-cols
---

::title::
マスクの種類

::left::
<span class="font-bold">Binary mask</span>

- 2値のマスク
- 0 か 1 で前景と背景を分ける
- 1つの対象だけを抜き出すときに便利

<br>

<span class="font-bold">Semantic mask</span>

- ピクセルごとにクラスを分類する
- 同じクラスの物体は同じ色になる
- 「人が何人いるか」は区別しない

::right::
<span class="font-bold">Instance mask</span>

- 同じクラスでも個体ごとに分ける
- 1人目と2人目を別のマスクにする
- 検出と分割を合わせたような考え方

<br>

```text
semantic: person person person
instance: person#1  person#2
```

::conc::
今回の基礎理解では、まず semantic segmentation を押さえる
