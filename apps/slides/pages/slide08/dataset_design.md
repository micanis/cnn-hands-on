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
  open_palm, x1, y1, x2, y2

image_002.jpg
  open_palm, x1, y1, x2, y2
  open_palm, x1, y1, x2, y2
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
- `0` は `open_palm` のクラスID
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

- 手のひらが開いて見えている手
- 指が開いていて、パーと判断できる手
- 片手でも両手でもよい
- 1枚に複数ある場合はすべて囲む

<br>

クラス名は必ず `open_palm`

::right::
<span class="font-bold">囲まないもの</span>

- グー
- チョキ
- 手の甲
- 指が閉じた手
- 判定が曖昧な手
- 人物全体や顔
- 背景の物体

<br>

パーがない画像は、何も囲まずに残す

::conc::
1クラス構成では「検出したいものだけを囲む」ことが重要

---

::title::
よいデータセットにするために

::default::
- **背景を変える**

  教室、廊下、机の前、白い壁など、同じ背景に偏らせない

- **距離を変える**

  近い手、遠い手、画面端にある手も入れる

- **向きを変える**

  正面、斜め、少し傾いた手を入れる

- **明るさを変える**

  明るい場所、少し暗い場所、影がある場所も入れる

- **負例を入れる**

  パーが写っていない画像も入れることで、何でも検出するモデルになりにくくする

::conc::
モデルはデータセットに含まれるパターンからしか学習できない
