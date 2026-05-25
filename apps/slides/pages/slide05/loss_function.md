---
layout: two-cols
---

::title::
損失関数とは

::left::
モデルの予測がどれだけ正解から<span class="font-bold">ズレている</span>かを数値化する関数

<span class="font-bold">なぜ必要か</span>

- モデルは最初ランダムな重みで予測 → 当然間違える
- **どれくらい間違えたか**を数値で測る必要がある
- この数値（損失）を**小さくする**のが学習の目標

::right::
<div class="flex flex-col items-center gap-2 h-full justify-center">
  <div class="text-xs font-bold text-gray-600">良い予測 → 正解クラスの確率が高い</div>
  <SoftmaxBarChart type="good" height="150px" />
  <div class="text-xs font-bold text-gray-600 mt-2">悪い予測 → 正解クラスの確率が低い</div>
  <SoftmaxBarChart type="bad" height="150px" />
</div>

::conc::
学習 = 損失関数の値を小さくすること

---
layout: two-cols
---

::title::
CrossEntropyLoss

::left::
<span class="font-bold"><span class="text-blue-400">分類タスク</span>の定番損失関数</span>

内部で2つの処理を行う:

1. **Softmax**: スコアを確率に変換

$p_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$

2. **負の対数尤度**: 正解クラスの確率が高いほど損失が小さい

$L = -\log(p_{\text{正解}})$


::right::
<div class="flex flex-col items-center justify-center h-full">
  <div class="text-xs font-bold text-gray-600 mb-1">-log(p) のグラフ</div>
  <CrossEntropyChart height="260px" />
</div>

::conc::
CrossEntropyLoss = Softmax + 負の対数尤度（PyTorchでは1行で書ける）

---
layout: two-cols
---

::title::
PyTorchでの損失計算

::left::
<span class="font-bold">nn.CrossEntropyLoss の使い方</span>

<Transform :scale="0.8">

```python
import torch
import torch.nn as nn

# 損失関数の定義
criterion = nn.CrossEntropyLoss()

# モデルの出力（バッチサイズ2、3クラス）
outputs = torch.tensor([
    [2.0, 1.0, 0.1],  # サンプル1
    [0.5, 2.5, 0.3],  # サンプル2
])

# 正解ラベル
labels = torch.tensor([0, 1])  # 猫, 犬

# 損失を計算
loss = criterion(outputs, labels)
print(f"Loss: {loss.item():.4f}")
```

</Transform>

::right::
<span class="font-bold">ポイント</span>

- `outputs`: Softmax**前**の生スコアを渡す
- `labels`: 正解のクラス番号（0, 1, 2...）
- PyTorchが内部でSoftmaxを計算してくれる

<span class="font-bold">注意</span>

<Transform :scale="0.9">

```python
# OK: 自分でSoftmaxをかけてはいけない
outputs = F.softmax(outputs, dim=1)
loss = criterion(outputs, labels)

# NG: 生スコアをそのまま渡す
loss = criterion(outputs, labels)
```

</Transform>

::conc::
CrossEntropyLossにはSoftmax前の値を渡す
