---
layout: two-cols
---

::title::
分類用データセットとの違い

::left::
<span class="font-bold">分類用データセット</span>

```text
image_001.jpg -> dog
image_002.jpg -> cat
image_003.jpg -> dog
```

<br>

- 画像ごとにラベルを付ける
- 画像全体が分類対象
- 犬猫分類やMNISTで使った形式

::right::
<span class="font-bold">検出用データセット</span>

```text
image_001.jpg
  face, x1, y1, x2, y2

image_002.jpg
  face, x1, y1, x2, y2
  face, x1, y1, x2, y2
```

<br>

- 1枚に複数の物体があってよい
- 物体ごとにクラスと位置を付ける
- 何も写っていない画像も使える

::conc::
検出データでは「どの画像か」だけでなく「画像のどこか」を記録する

---
layout: two-cols
---

::title::
バウンディングボックスとは

::left::
<span class="font-bold">物体を囲む四角形</span>

- 左上の座標
- 右下の座標
- または中心座標と幅・高さ

<br>

```text
x1, y1, x2, y2
```

<br>

検出モデルは、この四角形の位置も予測する

::right::
<span class="font-bold">YOLO形式の例</span>

```text
class_id x_center y_center width height
```

<br>

```text
0 0.52 0.41 0.24 0.35
```

<br>

- 座標は0から1に正規化される
- `0` は `face` のクラスID
- Roboflowが形式変換を行ってくれる

::conc::
人間がボックスを付け、モデルはそのボックスを再現できるように学習する

---
layout: two-cols
---

::title::
1クラス検出の考え方

::left::
<span class="font-bold">囲むもの</span>

- 正面から顔が見えている人物
- 目、鼻、口など顔の主要部分が見えている人物
- 1人でも複数人でもよい
- 1枚に複数ある場合はすべて囲む

<br>

クラス名は必ず `face`

::right::
<span class="font-bold">囲まないもの</span>

- 後頭部
- 顔がほとんど隠れている人物
- 横顔すぎて顔と判断しづらい人物
- 判定が曖昧な人物
- 人物全体や体
- 背景の物体

<br>

顔がない画像は、何も囲まずに残す

::conc::
1クラス構成では「検出したいものだけを囲む」ことが重要

---

::title::
よいデータセットにするために

::default::
- **背景を変える**

  教室、廊下、机の前、白い壁など、同じ背景に偏らせない

- **距離を変える**

  近い顔、遠い顔、画面端にある顔も入れる

- **向きを変える**

  正面、斜め、少し傾いた顔を入れる

- **明るさを変える**

  明るい場所、少し暗い場所、影がある場所も入れる

