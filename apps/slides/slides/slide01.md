---
theme: tut
layout: cover
---

::title::
CNNを用いた画像認識の仕組みと実装

::date::
2026年04月22日

::default::
山中春輝

---
layout: two-cols
---

::title::
授業計画

::left::
- <span class="text-blue-500" >第1回 ガイダンス / 犬猫の分類を試してみよう</span>
- 第2回 画像とは / 畳み込みの実装
- 第3回 活性化関数とは / プーリング層の実装
- 第4回 全結合層の実装 / CNNのモデル構築
- 第5回 損失関数とは / パラメータ更新の基礎
- 第6回 モデルの学習ループ実装と精度評価
- 第7回 転移学習の実装と精度評価
- 第8回 物体検出タスクの基礎及び推論

::right::
- 第9回 物体検出モデルの精度評価
- 第10回 領域分割タスクの基礎及び推論
- 第11回 データ拡張の重要性
- 第12回 エラー分析手法
- 第13回 総合演習（１）
- 第14回 総合演習（２）
- 第15回 総合演習（３）
- 第16回 総合演習（４）
  
::conc::
授業計画は変更される可能性があります

---
layout: toc
---

::title::
目次

---
layout: section
---

# 授業の概要

---


::title::
本授業の位置づけ

::default::
1. 現在主流の生成AI (ChatGPT等) の裏側を理解する

    → 中身の理解をし生成AIの限界や可能性を正しく見極める

2. 使う側から作る側へ
   
    → 個人・卒業制作のための土台として

3. AIシステム科にふさわしい素養のため

    → AIエンジニアとしての技術的素養を身につける

4. <span class="text-red-600 font-bold">単位は出ません</span>

---

::title::
学べるもの

::default::
<br>

- プログラミングを用いた画像の処理方法
  
<br>

- Pytorchを用いた機械学習プログラミング
  

<br>

- 機械学習モデル (CNN) の実装方法
  
---

::title::
必要なもの

::default::
<br>

- ノートPC
  
<br>

- Googleアカウント

---
layout: section
---

# 初回セットアップ

---
layout: two-cols
---

::title::
Google Colaboratory (Colab)

::left::
**メリット**
- 機械学習の定番ライブラリが<br>あらかじめインストール済み
- <span class="text-blue-500">無料枠でもGPUが利用可能</span>
- ブラウザのみで完結し、環境構築が不要

::right::
**デメリット**
- 連続実行時間や放置による制限がある<br>（セッション切れでデータがリセットされる）
- OSレベルの細かな環境構築が難しい
- UIのレスポンスが遅い（もっさりしている）
- Jupyter (.ipynb) ベースのため、<br>純粋な.pyファイルでの本格的な開発に不向き

::conc::
本格的なシステム開発には向かないが、誰でも同じ環境がすぐ作れるため授業には最適

---

::title::
Colabの使い方 (ハンズオン)

::default::

1. 授業資料一覧から`colab_setup.ipynb`というファイルをダウンロードする
2. Google Drive (https://accounts.google.com/Login?hl=ja&service=writely&lp=1) を開く
3. マイドライブ (MyDrive) フォルダを開いた状態で画面左上の`＋ボタン`を押す
4. `ファイルをアップロード`から先程ダウンロードした`colab_setup.ipynb`を選択する
5. アップロードが終わったら`colab_setup.ipynb`を開き、全てを実行する (権限はすべて許可)
6. もし開けない場合はGoogleドライブ内に`Google Colaboratory`をインストールする
7. その後、作成された`cnn-hands-on`を開き、`notebooks`を開く
8. `01_datasets.ipynb`を開き、全てを実行する (権限はすべて許可)
9. セットアップに使用した`colab_setup.ipynb`は次回移行も使うので`cnn-hands-on`へ移動させておく

<br>

<span class="text-2xl font-bold text-neutral-800">
    これにてセットアップは完了です！ お疲れ様でした
</span>

---
layout: section
---

# CNNについて

---

::title::
CNN（畳み込みニューラルネットワーク）とは

::default::

CNNは、人間の<span class="font-bold">視覚野の仕組み</span>をモデルにした、画像認識に特化したネットワーク

以下は画像分類タスクでの例

* **3つの主要なプロセス**
    1.  **畳み込み層**: 縦の線、横の線、特定の模様などを検出する
    2.  **プーリング層**: ズレや歪みの影響を抑え、データを扱いやすくする
    3.  **全結合層**: 見つけた特徴を組み合わせて「これは猫だ」と最終判断する

---

::title::
処理の流れ

::default::

入力された画像データが、どのように処理されて最終的な出力を得るのか

以下は画像分類タスクでの例

<div class="mt-8 p-4 bg-gray-50/50 rounded-xl border border-gray-200">
  <div class="flex items-center justify-between h-56 relative max-w-full">

  <v-click>
      <div class="flex flex-col items-center flex-1">
        <span class="text-xs font-bold mb-2 text-gray-500">1. 入力</span>
        <div class="relative w-30 h-30 border-2 border-gray-300 rounded bg-white p-1 shadow-sm">
          <img src="./public/slide01/cat.jpg" class="w-full h-full object-cover rounded-sm" />
        </div>
        <span class="mt-2 text-[10px] text-gray-400">画像データ</span>
      </div>
    </v-click>

  <div class="text-gray-300 text-xl font-bold">→</div>

  <v-click>
      <div class="flex flex-col items-center flex-1">
        <span class="text-xs font-bold mb-2 text-gray-500">2. 特徴抽出</span>
        <div class="relative w-30 h-30 flex items-center justify-center">
          <div v-for="i in 4" :key="i" 
               class="absolute w-12 h-12 border border-blue-400 bg-blue-100/80 rounded shadow-sm"
               :style="{ transform: `translate(${(i-1)*4}px, ${(i-1)*-4}px)` }">
          </div>
        </div>
        <span class="mt-2 text-[10px] text-gray-400 text-center">畳み込み・プーリング</span>
      </div>
    </v-click>

  <div class="text-gray-300 text-xl font-bold">→</div>

  <v-click>
      <div class="flex flex-col items-center flex-1">
        <span class="text-xs font-bold mb-2 text-gray-500">3. 分類</span>
        <div class="w-22 h-30 flex items-center justify-center bg-green-50 rounded-lg border border-green-200">
          <div class="grid grid-cols-2 gap-1">
            <div v-for="i in 8" :key="i" class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          </div>
        </div>
        <span class="mt-2 text-[10px] text-gray-400 text-center">全結合層</span>
      </div>
    </v-click>

  <div class="text-gray-300 text-xl font-bold">→</div>

  <v-click>
      <div class="flex flex-col items-center flex-1">
        <span class="text-xs font-bold mb-2 text-gray-500">4. 出力</span>
        <div class="w-38 h-30 bg-white border border-orange-300 rounded shadow-sm p-1.5 flex flex-col justify-center gap-1">
          <div class="flex justify-between items-center bg-orange-100 px-1 rounded-sm">
            <span class="text-[10px] font-bold">ネコ</span>
            <span class="text-[9px]">98%</span>
          </div>
          <div class="flex justify-between items-center px-1">
            <span class="text-[10px]">イヌ</span>
            <span class="text-[9px] text-gray-400">1.5%</span>
          </div>
        </div>
        <span class="mt-2 text-[10px] text-gray-400">最終判定</span>
      </div>
    </v-click>

  </div>
</div>

<style scoped>
.cnn-flow-container {
  perspective: 1000px;
}
.cnn-step {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.box-label {
  font-weight: bold;
  font-size: 0.9rem;
  margin-bottom: 10px;
  height: 2.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
}
.arrow::after {
  content: '→';
  position: absolute;
  top: 50%;
  right: -40px;
  transform: translateY(-50%);
  font-size: 2rem;
  color: #a1a1aa; /* text-neutral-400 */
}
</style>

---
layout: section
---

# 画像分類について

---
layout: two-cols
---

::title::
画像分類とは

::left::
<div class="w-full h-full flex justify-center">
    <img src="./public/slide01/cat.jpg">
</div>

::right::

Q. 左の画像は"犬" or "猫"どちらですか？

A. <span v-click class="text-2xl">猫</span>

<div class="mt-10" />

<v-click>
<span class="font-bold">画像分類 (Classification) とは</span>

入力された画像が、あらかじめ定義された

どのクラス (ラベル) に属するかを予測するタスク
</v-click>

::conc::
人間には簡単でも、コンピュータには長年の難問だった

---

::title::
CNNによる画像分類のデモ

::default::
<ClassificationDemo />

---
layout: two-cols
---

::title::
どのように分類するか

::left::
<br>
 
**CNN登場より前（従来手法）**
- 人間が頑張って「特徴」を定義していた
- 「耳が尖っている」「毛の模様が…」といったルールを数式化してプログラム
- 職人技が必要で、精度にも限界があった

::right::
<br>

**CNN登場後（ディープラーニング）**
- 大量の画像データと正解を丸投げする
- コンピュータが勝手に「犬っぽさ」「猫っぽさ」の**特徴を自力で見つけ出す**
- これにより精度が爆発的に向上した

---
layout: three-cols
---

::title::
今回のコード一覧

::left::
<Transform :scale="0.8">

<<< ../code/slide01/01_intro.py#part1

</Transform>

::center::
<Transform :scale="0.8">

<<< ../code/slide01/01_intro.py#part2

</Transform>

::right::
**何をしているか**

1. ライブラリを読み込む
2. 学習済みモデルの設定
3. 推論用画像の設定
4. ランダムに画像１枚を推論する
5. 自由にアップロードされた画像を推論する

---
layout: section
---

# まとめ

---

::title::
本日のまとめ

::default::

- **目的の共有**
  
  AIを作る側のエンジニアを目指す
  

- **環境構築の完了**

  Colabを用いて、ブラウザ一つでいつでもGPUを使った機械学習ができる環境が整った


- **画像分類とCNNの凄さ**

  人間がルールを教え込むのではなく、大量のデータからコンピュータ自身に<br><span class="text-blue-500">特徴を見つけ出させる</span>のがディープラーニング (CNN)


- **まずは動かしてみた**

  学習済みのモデルを使って、実際にAIが画像を分類する推論プロセスを体験した

---
src: ../pages/slide01/appendix.md
---
