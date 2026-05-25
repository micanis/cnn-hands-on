---
layout: two-cols
---

::title::
学習の全体像

::left::
<span class="font-bold">エポック × バッチの2重ループ</span>

<Transform :scale="0.85">

```python
for epoch in range(num_epochs):
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

</Transform>


**エポック (Epoch)**: 全データを1周すること

**バッチ (Batch)**: 1回の更新で使うデータの塊

::right::
<div class="flex flex-col items-center justify-center h-full">
  <div class="text-sm font-bold mb-3">学習の流れ</div>
  <div class="flex flex-col gap-1 items-center text-xs w-full px-4">
    <div class="w-full px-3 py-2 bg-purple-100 border border-purple-300 rounded text-center font-bold">エポック 1 / 10</div>
    <div class="w-5/6 flex flex-col gap-1 items-center">
      <div class="w-full px-2 py-1 bg-blue-50 border border-blue-200 rounded text-center">バッチ 1: 画像64枚 → 学習</div>
      <div class="w-full px-2 py-1 bg-blue-50 border border-blue-200 rounded text-center">バッチ 2: 画像64枚 → 学習</div>
      <div class="w-full px-2 py-1 bg-gray-50 border border-gray-200 rounded text-center">... (約938バッチ)</div>
    </div>
    <span>↓</span>
    <div class="w-full px-3 py-2 bg-green-100 border border-green-300 rounded text-center">検証 (Validation) → 精度チェック</div>
    <span>↓</span>
    <div class="w-full px-3 py-2 bg-purple-100 border border-purple-300 rounded text-center font-bold">エポック 2 / 10 ...</div>
  </div>
</div>

::conc::
エポックを重ねるごとに、モデルの精度が上がっていく

---
layout: two-cols
---

::title::
Train vs Validation

::left::
<span class="font-bold">なぜ分ける必要があるのか</span>

**訓練 (Train)**
- モデルの重みを**更新する**
- `model.train()` モードで実行

**検証 (Validation)**
- モデルの性能を**確認する**
- `model.eval()` モードで実行
- **重みは更新しない**

::right::
<span class="font-bold">過学習 (Overfitting)</span>

教科書を丸暗記して、模試で解けない状態

Train Loss↓ だが Val Loss↑ なら過学習

::conc::
Train Lossだけでなく、Validation Lossも監視することが重要

---
layout: two-cols
---

::title::
学習曲線の読み方

::left::
<div class="flex flex-col h-full">
  <div class="text-xs font-bold text-center mb-1">OK: 良い学習曲線</div>
  <LossCurveChart type="good" height="200px" />
  <div class="text-xs text-gray-500 text-center mt-1">
    Train/Val Loss 共に順調に下降
  </div>
</div>

::right::
<div class="flex flex-col h-full">
  <div class="text-xs font-bold text-center mb-1 text-red-500">NG: 過学習の例</div>
  <LossCurveChart type="overfit" height="200px" />
  <div class="text-xs text-gray-500 text-center mt-1">
    Train Loss↓ だが Val Loss↑ → 過学習!
  </div>
</div>

::conc::
学習曲線を見て、過学習していないか常にチェックする

---

::title::
実演

::default::
<div class="h-full w-full flex items-center justify-center"><h2>Colabで見てみよう!</h2></div>
