---
layout: two-cols
---

::title::
勾配降下法とは

::left::
損失関数の値を**小さくする**ようにパラメータ（重み）を更新する手法

$W \leftarrow W - \eta \cdot \frac{\partial L}{\partial W}$

- $W$: パラメータ（重み）
- $\eta$: 学習率（どれだけ移動するか）
- $\frac{\partial L}{\partial W}$: 勾配（坂の傾き）

::right::
<div class="flex flex-col items-center justify-center h-full">
  <GradientDescentChart height="320px" />
</div>

::conc::
勾配降下法は「坂を下って谷底を探す」最適化手法

---
layout: two-cols
---

::title::
学習率とオプティマイザ

::left::
<span class="font-bold">1回の更新でどれだけ動くかのハイパーパラメータ</span>

| 学習率 | 特徴 |
|---|---|
| **大きすぎる** | 発散する（谷を飛び越える） |
| **小さすぎる** | 収束が遅い（なかなか進まない） |
| **適切** | 効率的に最小値に到達 |

一般的な初期値: `0.001`

::right::
<span class="font-bold">代表的なオプティマイザ</span>

<div class="mt-2 text-sm">

**SGD** (確率的勾配降下法)
```python
optimizer = optim.SGD(
    model.parameters(), lr=0.01
)
```

**Adam** (適応的学習率) ← おすすめ
```python
optimizer = optim.Adam(
    model.parameters(), lr=0.001
)
```

</div>

<div class="mt-4 text-sm">
Adamは学習率を自動調整してくれるため、初心者にも扱いやすい
</div>

::conc::
今回はAdam (lr=0.001) を使用する

---
layout: two-cols
---

::title::
学習の4ステップ

::left::
<span class="font-bold">1回のバッチ処理の流れ</span>

<Transform :scale="0.9">

```python
for images, labels in train_loader:
    # ① 勾配のリセット
    optimizer.zero_grad()

    # ② 予測（順伝播）
    outputs = model(images)

    # ③ 損失の計算
    loss = criterion(outputs, labels)

    # ④ 勾配の計算（逆伝播）
    loss.backward()

    # ⑤ パラメータの更新
    optimizer.step()
```

</Transform>

::right::
<div class="flex flex-col items-center justify-center h-full">
  <div class="flex flex-col gap-2 items-center text-xs">
    <div class="px-4 py-2 bg-gray-100 border rounded font-bold">① 勾配リセット</div>
    <span>↓</span>
    <div class="px-4 py-2 bg-blue-100 border rounded font-bold">② 順伝播 (Forward)</div>
    <span>↓</span>
    <div class="px-4 py-2 bg-orange-100 border rounded font-bold">③ 損失計算 (Loss)</div>
    <span>↓</span>
    <div class="px-4 py-2 bg-red-100 border rounded font-bold">④ 逆伝播 (Backward)</div>
    <span>↓</span>
    <div class="px-4 py-2 bg-green-100 border rounded font-bold">⑤ パラメータ更新 (Step)</div>
  </div>
  <div class="mt-4 text-xs text-gray-500">これを全バッチ分繰り返す = 1エポック</div>
</div>

::conc::
この5ステップがディープラーニング学習の基本パターン
