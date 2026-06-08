---
layout: two-cols
---

::title::
第六回コードからの変更点

::left::
<span class="font-bold">そのまま使う部分</span>

```python
train_loader, val_loader, test_loader = get_dc_dataloaders(
    data_dir=LOCAL_DATA_DIR,
    data_size="small",
    batch_size=32
)

criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    # Train
    # Validation
    # Best Model Save
```

::right::
<span class="font-bold">差し替える部分</span>

```python
# 自作ResNet
model = ResNet18(num_classes=2).to(device)

# 転移学習
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
```

<br>

- DataLoaderと学習ループはほぼ同じ
- モデルを事前学習済みResNet18にする
- 最後の分類層だけを犬猫用にする

::conc::
「構造を自作する」から「学習済みの重みを使う」へ進む

---

::title::
必要なimport

::default::
<Transform :scale="0.95">

```python
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
```

</Transform>

<br>

- `models`: torchvisionに用意されている代表的なモデルを使う

- `ResNet18_Weights`: ResNet18の事前学習済み重みを指定する

- `weights.transforms()` を使うと、事前学習時と同じ前処理を使える

---
layout: two-cols
---

::title::
前処理を合わせる

::left::
<span class="font-bold">自作ResNetの前処理</span>

```python
transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

::right::
<span class="font-bold">転移学習の前処理</span>

```python
weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()
```

<br>

- 画像サイズ
- Tensor変換
- 正規化

を事前学習済みモデルに合わせる

::conc::
学習済みモデルを使うときは、入力の前処理も合わせる

---

::title::
事前学習済みResNet18を読み込む

::default::
<Transform :scale="0.95">

```python
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
```

</Transform>

<br>

- `DEFAULT` は推奨される事前学習済み重みを使う指定

- 初回実行時は重みファイルのダウンロードが発生する

- Colabではインターネット接続が必要

---
layout: two-cols
---

::title::
最終層を差し替える

::left::
<span class="font-bold">元のResNet18</span>

```python
print(model.fc)
```

```text
Linear(in_features=512,
       out_features=1000)
```

ImageNetの1000クラス分類用

::right::
<span class="font-bold">犬猫分類用に変更</span>

```python
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)
model = model.to(device)
```

<br>

- 出力を2クラスにする
- `CrossEntropyLoss` を使うためSoftmaxは入れない

::conc::
最終層だけを変えれば、第六回と同じ学習ループで学習できる

---
layout: two-cols
---

::title::
最終層だけ学習する

::left::
<span class="font-bold">特徴抽出器を固定する</span>

```python
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
```

::right::
<span class="font-bold">optimizerはfcだけ</span>

```python
optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)
```

<br>

- 学習するパラメータが少ない
- smallデータでも安定しやすい
- 自作ResNetより高精度になりやすい

::conc::
最初は「特徴抽出器を固定し、分類器だけ学習」が扱いやすい

---
layout: two-cols
---

::title::
精度を比較する

::left::
<span class="font-bold">自作ResNet18</span>

- ランダム初期化から学習
- 犬猫データだけで特徴を学ぶ
- `data_size="large"` で学習
- 目安: 85-90%前後

::right::
<span class="font-bold">転移学習</span>

- ImageNet学習済み重みを利用
- 画像特徴を再利用
- `data_size="small"` でも高精度を狙える
- 目安: 98-100%前後

::conc::
構造の改善より、事前学習済み特徴の再利用が大きく効くことを確認する

---

::title::
転移学習を実装してみよう

::default::
- `07_transfer_learning.ipynb` を開く

- `models.resnet18(weights=weights)` で事前学習済みモデルを読み込む

- `model.fc` を2クラス分類用に差し替える

- 特徴抽出器を固定し、`model.fc` だけ学習する

- `data_size="small"`、`num_epochs=5` で実行する

<br>

<span class="text-xl font-bold">Colabで `07_transfer_learning.ipynb` を実装してみましょう</span>
