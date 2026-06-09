---
layout: two-cols
---

::title::
Roboflowで行う作業

::left::
<span class="font-bold">データセット作成の流れ</span>

1. プロジェクトを作成する
2. クラス `open_palm` を登録する
3. 画像をアップロードする
4. バウンディングボックスを付ける
5. Train / Valid / Test に分割する
6. YOLO形式でエクスポートする

::right::
<span class="font-bold">今回の目標</span>

- クラス数: 1
- クラス名: `open_palm`
- 画像枚数: 約200枚
- 形式: Object Detection
- 出力: YOLO用データセット

<br>

生徒全員で1つのデータセットを作る

::conc::
Roboflowを使うと、画像管理、アノテーション、形式変換をまとめて行える

---

::title::
撮影ルール

::default::
- **パーの手を写す**

  手のひらが開いて見える画像を多めに集める

- **同じ構図に偏らせない**

  背景、距離、角度、明るさを変える

- **手以外も自然に写ってよい**

  顔、服、机、教室などが写っていてもよい

- **負例も入れる**

  グー、チョキ、手の甲、手が写っていない画像も少し入れる

- **個人情報に注意する**

  不要な顔、名札、画面、個人が特定される情報は写さないようにする

::conc::
同じような画像ばかりだと、実際のWebカメラ画像でうまく動きにくい

---
layout: two-cols
---

::title::
アノテーションルール

::left::
<span class="font-bold text-blue-500">囲む</span>

- 開いた手のひら
- 指が開いていてパーと判断できる手
- 画面内に複数ある場合はすべて
- 少し傾いていてもパーなら囲む

<br>

できるだけ手の範囲にぴったり合わせる

::right::
<span class="font-bold text-red-500">囲まない</span>

- グー
- チョキ
- 手の甲
- 指が閉じた手
- 判定が曖昧な手
- 腕全体
- 人物全体
- 背景の物体

<br>

パーがない画像は何も囲まない

::conc::
アノテーションのルールが揃うほど、モデルは学習しやすくなる

---
layout: two-cols
---

::title::
よいボックスと悪いボックス

::left::
<span class="font-bold">よい例</span>

<div class="relative w-72 h-44 border-2 border-gray-300 bg-gray-50 mx-auto">
  <div class="absolute left-24 top-8 w-24 h-28 border-4 border-blue-500"></div>
  <div class="absolute left-24 top-2 px-2 py-1 bg-blue-500 text-white text-sm font-bold">open_palm</div>
</div>

<br>

- 手の範囲に近い
- 余白が少ない
- 対象が明確

::right::
<span class="font-bold">悪い例</span>

<div class="relative w-72 h-44 border-2 border-gray-300 bg-gray-50 mx-auto">
  <div class="absolute left-10 top-5 w-48 h-36 border-4 border-red-500"></div>
  <div class="absolute left-10 top-0 px-2 py-1 bg-red-500 text-white text-sm font-bold">open_palm</div>
</div>

<br>

- 腕や背景を大きく含む
- 手の位置が曖昧
- 学習したい範囲がぶれる

::conc::
ボックスは「モデルに見つけてほしい範囲」を教える教師データ
