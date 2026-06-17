---
layout: two-cols
---

::title::
領域分割とは

::left::

<span class="font-bold">領域分割</span>

- ピクセルごとにクラスを決める
- 物体の輪郭に沿って塗り分ける
- 例: 人の輪郭、道路、空、背景を分ける

::right::
<div class="relative w-80 h-48 mx-auto border-2 border-gray-300 bg-gray-50 rounded-sm overflow-hidden">
  <div class="absolute inset-0 grid grid-cols-8 grid-rows-4 gap-0.5 p-2">
    <div class="col-span-3 row-span-3 bg-blue-400/60 rounded-sm"></div>
    <div class="col-span-2 row-span-2 bg-emerald-400/60 rounded-sm"></div>
    <div class="col-span-3 row-span-4 bg-amber-300/60 rounded-sm"></div>
  </div>
  <div class="absolute left-3 top-3 px-2 py-1 bg-blue-600 text-white text-sm font-bold rounded">person</div>
  <div class="absolute right-4 top-6 px-2 py-1 bg-emerald-600 text-white text-sm font-bold rounded">road</div>
  <div class="absolute right-8 bottom-4 px-2 py-1 bg-amber-600 text-white text-sm font-bold rounded">sky</div>
</div>

::conc::
領域分割は、検出よりも細かく「形」を扱えるタスク

---
layout: two-cols
---

::title::
領域分割でわかること

::left::
<span class="font-bold">何に使うか</span>

- 自動運転の道路認識
- 医用画像の病変領域抽出
- 背景削除や人物切り抜き
- 作物や土地の領域推定

<br>

領域分割は「何があるか」だけでなく、<span class="text-blue-500 font-bold">その輪郭</span>を知りたいときに役立つ

::right::
<span class="font-bold">検出との違い</span>

```text
検出:  [ 人物 ]  [ 人物 ]
分割:  人物人物人物背景背景
       人物人物人物背景背景
       背景背景人物人物人物
```

<br>

- 検出は四角形で近似する
- 分割は境界をより細かく表現する

::conc::
物体の位置だけでなく、境界の精密さが重要な場面で領域分割を使う

