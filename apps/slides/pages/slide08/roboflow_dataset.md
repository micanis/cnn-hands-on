---
layout: two-cols
---

::title::
Roboflowで行う作業

::left::
<span class="font-bold">データセット作成の流れ</span>

1. [プロジェクトに参加する](https://app.roboflow.com/join/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3Jrc3BhY2VJZCI6ImVEQnQ4VGNQVzdQTUZVNFpBS0ZvZ2duTkVuajEiLCJyb2xlIjoib3duZXIiLCJpbnZpdGVyIjoieWFtYW5ha2FoYXJ1a2lfaXRfc2FAZy5uZWVjLmFjLmpwIiwiaWF0IjoxNzgxMDczMDg5fQ.MJUEqv85K0mpCSD3a_kBKDf-YILXZQaBQ2TyNfO1JE4)
2. クラス `face` を登録する
3. 画像をアップロードする
4. バウンディングボックスを付ける
5. Train / Valid / Test に分割する
6. YOLO形式でエクスポートする

::right::
<span class="font-bold">今回の目標</span>

- クラス数: 1
- クラス名: `face`
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
- **顔を写す**

  顔がはっきり見える画像を多めに集める

- **同じ構図に偏らせない**

  背景、距離、角度、明るさを変える

- **顔以外も自然に写ってよい**

  服、机、教室などが写っていてもよい

- **負例も入れる**

  後ろ姿、顔が隠れた人物、人物が写っていない画像も少し入れる

---
layout: two-cols
---

::title::
アノテーションルール

::left::
<span class="font-bold text-blue-500">囲む</span>

- 正面から見えている顔
- 目、鼻、口など顔の主要部分が見えている顔
- 画面内に複数ある場合はすべて
- 少し傾いていても顔と判断できれば囲む

<br>

できるだけ顔の範囲にぴったり合わせる

::right::
<span class="font-bold text-red-500">囲まない</span>

- 後頭部
- 顔がほとんど隠れている人物
- 横顔すぎて顔と判断しづらい人物
- 判定が曖昧な人物
- 人物全体
- 背景の物体

<br>

顔がない画像は何も囲まない

::conc::
アノテーションのルールが揃うほど、モデルは学習しやすくなる
