---
layout: default
---

::title::
検出 (Detection) とは

::default::
<div class="grid grid-cols-2 gap-8 text-xl">
  <div>
    <div class="font-bold mb-3">分類</div>
    <ul>
      <li>画像全体に1つのラベルを付ける</li>
      <li>何が写っているかを答える</li>
      <li>例: この画像は犬</li>
    </ul>
  </div>
  <div>
    <div class="font-bold mb-3">検出</div>
    <ul>
      <li>物体ごとにラベルと位置を出す</li>
      <li>何がどこにあるかを答える</li>
      <li>例: 顔がこの範囲にある</li>
    </ul>
  </div>
</div>

<br>

<span class="text-xl font-bold">物体検出は、クラス分類に「位置の予測」を加えたタスク</span>

---
layout: two-cols
---

::title::
検出結果の見方

::left::
<span class="font-bold">検出モデルの出力</span>

- `class`

  検出した物体の種類

- `confidence`

  その予測の確信度

- `bounding box`

  物体の位置を表す四角形

::right::
<span class="font-bold">今回の例</span>

```text
class: face
confidence: 0.83
box: x1=150, y1=70, x2=310, y2=260
```

<br>

<div class="relative w-72 h-44 border-2 border-gray-300 bg-gray-50 mx-auto">
  <div class="absolute left-20 top-7 w-28 h-28 border-4 border-blue-500"></div>
  <div class="absolute left-20 top-1 px-2 py-1 bg-blue-500 text-white text-sm font-bold">face 0.83</div>
</div>


---

::title::
今回の検出対象

::default::
<div class="text-2xl leading-relaxed">

今回作るモデルは、人物の顔を検出します。

</div>

<br>

- 検出クラスは <span class="font-bold text-blue-500">`face`</span> の1つ

- 顔だけをバウンディングボックスで囲む

- 後頭部、横顔すぎる顔、顔がほとんど見えない人物は囲まない

- 顔が写っていない画像も、負例としてデータセットに入れる

<br>

<span class="text-xl font-bold">ゴール: Webカメラ画像から `face` を検出する</span>
