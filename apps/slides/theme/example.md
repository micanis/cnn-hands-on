---
theme: ./
layout: cover
---

::header::
東京工科大学学生用スライドテンプレート

::date::
2026年04月06日

G2126038 山中春輝

---
layout: toc
---

::header::
目次

---
layout: section
---

# 代表的な生成AIの比較

---
layout: default
---

::header::
現在主流のチャット型AI(default layout)

::default::
現在、ビジネスや日常で広く使われている代表的なAIツールは以下の4つです。
それぞれの強みを理解し、目的に合わせて使い分けることが重要です。

- **ChatGPT (OpenAI)**：世界的なAIブームの火付け役。万能で自然な対話が得意。
- **Gemini (Google)**：検索エンジンや各種Googleアプリとの連携が強力。
- **Claude (Anthropic)**：極めて自然な日本語と、長文の読み込み・要約に強い。
- **Copilot (Microsoft)**：WindowsやOffice製品に組み込まれた、最も身近なAI。

---
layout: two-cols
---

::header::
2大AIツールの比較(tow-cols layout)

::left::
**ChatGPT**

- **知名度No.1のパイオニア**
  世界にAIを普及させた立役者。
- **圧倒的な汎用性と対話力**
  人間らしく自然な会話が得意で、自分専用のAI（GPTs）を作ることも可能。

::right::
**Gemini**

- **Googleの強力なインフラ**
  ドキュメントやGmailなどとシームレスに連動。
- **検索に強い**
  最新の情報をGoogle検索から瞬時に集めて回答するのが得意。

::conc::
「拡張性のChatGPT」か、「最新情報と連携のGemini」かで使い分けるのがオススメ！

---
layout: three-cols
---

::header::
3大ツールの特徴（+Claude）(three-cols layout)

::left::
**ChatGPT**

- OpenAIが開発
- 圧倒的なユーザー数
- プラグインなど拡張機能が豊富で、どんな用途にも使える万能型

::center::
**Claude**

- Anthropicが開発
- 人間と見間違うほど自然な日本語
- 長い文章の処理や、プログラミングコードの生成に非常に優秀

::right::
**Gemini**

- Googleが開発
- 最新情報の検索と要約に強い
- 無料枠でも処理速度が速く、制限が緩いため日常使いに最適

::conc::
日常使いは「Gemini」、執筆や開発は「Claude」、万能アシスタントは「ChatGPT」！

---
layout: two-rows
---

::header::
得意領域の違い(two-rows layout)

::top::
**ChatGPT：クリエイティブな対話と拡張性**
世界初の対話型AIとして登場以来、圧倒的なデータ量で自然な対話を可能にしています。
文章作成の壁打ち相手や、アイデア出しのパートナーとして非常に優秀です。

::bottom::
**Gemini：リアルタイムな情報収集と作業効率化**
Googleの強みを活かし、今日のニュースや最新トレンドを瞬時にまとめてくれます。
表計算ソフトなどのビジネスツールとの相性も良く、日々の業務を大きく効率化します。

::conc::
どちらも日々進化を続けているため、まずは両方触ってみて好みの相棒を見つけましょう。

---
layout: section
---

# 様々なレイアウト作成

---
layout: bento
cols: 3
rows: 2
sizeA: w2 h2
---

::header::
いろいろな分割例 その1：メイン図解＋右サイドバー

::a::
<div class="w-full h-full bg-blue-200 rounded-lg flex items-center justify-center font-bold text-2xl">A (w2 h2)</div>

::b::
<div class="w-full h-full bg-green-200 rounded-lg flex items-center justify-center font-bold text-2xl">B (w1 h1)</div>

::c::
<div class="w-full h-full bg-emerald-200 rounded-lg flex items-center justify-center font-bold text-2xl">C (w1 h1)</div>

---
layout: bento
cols: 3
rows: 2
sizeA: w3 h1
---

::header::
いろいろな分割例 その2：トップ見出し＋3カラム

::a::
<div class="w-full h-full bg-purple-200 rounded-lg flex items-center justify-center font-bold text-2xl">A (w3 h1) : 概要・結論</div>

::b::
<div class="w-full h-full bg-orange-200 rounded-lg flex items-center justify-center font-bold text-2xl">B (w1 h1) : 手法1</div>

::c::
<div class="w-full h-full bg-amber-200 rounded-lg flex items-center justify-center font-bold text-2xl">C (w1 h1) : 手法2</div>

::d::
<div class="w-full h-full bg-yellow-200 rounded-lg flex items-center justify-center font-bold text-2xl">D (w1 h1) : 手法3</div>

---
layout: bento
cols: 3
rows: 2
sizeA: w1 h2
sizeB: w2 h2
---

::header::
いろいろな分割例 その3：左サイドメニュー＋メイン

::a::
<div class="w-full h-full bg-rose-200 rounded-lg flex items-center justify-center font-bold text-2xl">A (w1 h2)<br>前提条件など</div>

::b::
<div class="w-full h-full bg-pink-200 rounded-lg flex items-center justify-center font-bold text-2xl">B (w2 h2)<br>メインコンテンツ</div>

---
layout: bento
cols: 2
rows: 2
---

::header::
いろいろな分割例 その4：4象限（2x2グリッド）


::a::
<div class="w-full h-full bg-cyan-200 rounded-lg flex items-center justify-center font-bold text-2xl">A (自動)</div>

::b::
<div class="w-full h-full bg-sky-200 rounded-lg flex items-center justify-center font-bold text-2xl">B (自動)</div>

::c::
<div class="w-full h-full bg-blue-200 rounded-lg flex items-center justify-center font-bold text-2xl">C (自動)</div>

::d::
<div class="w-full h-full bg-indigo-200 rounded-lg flex items-center justify-center font-bold text-2xl">D (自動)</div>

---
layout: bento
cols: 4
rows: 2
sizeA: w1 h2
sizeB: w2 h1
sizeC: w1 h1
sizeD: w2 h1
sizeE: w1 h1
---

::header::
いろいろな分割例 その5：複雑なBentoスタイル

::a::
<div class="w-full h-full bg-slate-300 rounded-lg flex items-center justify-center font-bold text-2xl">A (w1 h2)</div>

::b::
<div class="w-full h-full bg-gray-300 rounded-lg flex items-center justify-center font-bold text-2xl">B (w2 h1)</div>

::c::
<div class="w-full h-full bg-zinc-300 rounded-lg flex items-center justify-center font-bold text-2xl">C (w1 h1)</div>

::d::
<div class="w-full h-full bg-neutral-300 rounded-lg flex items-center justify-center font-bold text-2xl">D (w2 h1)</div>

::e::
<div class="w-full h-full bg-stone-300 rounded-lg flex items-center justify-center font-bold text-2xl">E (w1 h1)</div>

---
layout: section
---

# 結論

---
layout: default
---

::header::
結論

::default::
<div class="w-full h-full flex justify-center items-center">
  <p class="text-3xl font-bold text-neutral-900">
    Slidevは自由自在な記述ができることが強み！！
  </p>
</div>