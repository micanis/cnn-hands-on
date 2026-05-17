---
layout: section
---

# 付録

---
layout: two-cols
---

::header::
コード解説①

::left::

<CodePane>

<<< ../../code/slide01/01_intro.py#cell02

</CodePane>

::right::
何をしているか

- あらかじめ作成されたCNNの構造を読み込む
- 学習済みモデルを読み込み、設定
- モデルを評価用に設定
- クラス　(ラベル) を定義する

---
layout: two-cols
---

::header::
コード解説②

::left::

<CodePane>

<<< ../../code/slide01/01_intro.py#cell03

</CodePane>

::right::
何をしているか

- データセットを読み込む
- 評価用のデータセットだけを取得する
- どのくらいの個数があるかの確認をする

---
layout: two-cols
---

::header::
コード解説③

::left::

<CodePane>

<<< ../../code/slide01/01_intro.py#cell04

</CodePane>

::right::
何をしているか

- ランダムにデータを選択する
- データから画像とクラス (ラベル) を読み取る
- 正解クラスを定義する
- pythonの画像描画ライブラリで図表を作成
- 入力画像を推論する
- 結果を出力する

---
layout: two-cols
---

::header::
コード解説④

::left::

<CodePane>

<<< ../../code/slide01/01_intro.py#cell05

</CodePane>

::right::
何をしているか

- Google Colabのファイルをアップロードできるように設定
- あとはコード③と同じ