---
layout: two-cols
---

::title::
代表的な物体検出モデル

::left::
<span class="font-bold">R-CNN系</span>

- 物体がありそうな領域を探す
- その領域を分類する
- 精度を重視したモデルが多い
- 処理は比較的重くなりやすい

<br>

```text
候補領域 -> 分類 -> 位置調整
```

::right::
<span class="font-bold">SSD / YOLO系</span>

- 画像全体から一度に検出する
- 推論が速い
- リアルタイム処理に向いている
- Webカメラ検出と相性がよい

<br>

```text
画像 -> クラス + ボックス
```

::conc::
検出モデルには、精度重視のものと速度重視のものがある

---
layout: two-cols
---

::title::
YOLOを使う理由

::left::
<span class="font-bold">YOLOの特徴</span>

- You Only Look Once の略
- 画像を一度見るだけで検出する
- 推論が高速
- 小さなモデルならColabでも扱いやすい
- 学習、推論、可視化のコードが短い

::right::
<span class="font-bold">今回の目的に合う点</span>

- Webカメラ画像で動かしやすい
- Roboflowのデータセットと接続しやすい
- `open_palm` だけの小さな検出モデルを作りやすい
- 結果をその場で画像に描画できる

::conc::
今回はモデル内部の細部より、データセット作成から推論までの流れを重視する

---

::title::
学習済みモデルと自作データセット

::default::
<div class="grid grid-cols-2 gap-8 text-xl">
  <div>
    <div class="font-bold mb-3">一般的な学習済みモデル</div>
    <ul>
      <li>人、車、犬、猫などを検出できる</li>
      <li>大規模データで学習済み</li>
      <li>すぐに推論できる</li>
    </ul>
  </div>
  <div>
    <div class="font-bold mb-3">今回作るモデル</div>
    <ul>
      <li>`open_palm` を検出する</li>
      <li>自分たちで画像を集める</li>
      <li>自分たちでボックスを付ける</li>
    </ul>
  </div>
</div>

<br>

<span class="text-xl font-bold">既存モデルにないクラスは、自分たちでデータセットを作って学習する</span>
