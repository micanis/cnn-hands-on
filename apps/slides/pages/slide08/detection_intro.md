---
layout: two-cols
---

::title::
分類と検出の違い

::left::
<span class="font-bold">画像分類</span>

```text
Input:  画像
Output: クラス
```

<br>

- 画像全体に1つのラベルを付ける
- 犬か猫か、数字が何かを判定する
- 物体の位置は扱わない

<br>

```text
dog: 0.92
cat: 0.08
```

::right::
<span class="font-bold">物体検出</span>

```text
Input:  画像
Output: クラス + 位置
```

<br>

- 画像内の物体を探す
- それぞれの物体のクラスを判定する
- 物体の位置を四角形で表す

<br>

```text
open_palm: 0.87, box=(120, 80, 260, 250)
```

::conc::
分類は「何が写っているか」、検出は「どこに何があるか」を扱う

---
layout: two-cols
---

::title::
検出結果の3つの情報

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
class: open_palm
confidence: 0.83
box: x1=150, y1=70, x2=310, y2=260
```

<br>

<div class="relative w-72 h-44 border-2 border-gray-300 bg-gray-50 mx-auto">
  <div class="absolute left-20 top-7 w-28 h-28 border-4 border-blue-500"></div>
  <div class="absolute left-20 top-1 px-2 py-1 bg-blue-500 text-white text-sm font-bold">open_palm 0.83</div>
</div>

::conc::
検出では「何か」だけでなく、画像内の位置も学習する

---

::title::
今回の検出対象

::default::
<div class="text-2xl leading-relaxed">

今回作るモデルは、人物の開いた手のひらを検出します。

</div>

<br>

- 検出クラスは <span class="font-bold text-blue-500">`open_palm`</span> の1つ

- パーの手だけをバウンディングボックスで囲む

- グー、チョキ、手の甲、曖昧な手は囲まない

- パーが写っていない画像も、負例としてデータセットに入れる

<br>

<span class="text-xl font-bold">ゴール: Webカメラ画像から `open_palm` を検出する</span>
