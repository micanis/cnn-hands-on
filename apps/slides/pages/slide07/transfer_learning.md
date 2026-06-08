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
# 第六回
model = SimpleCNN().to(device)

# 第七回
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
```

<br>

- データセットの読み込みは同じ
- 損失関数も同じ
- 学習ループも同じ
- モデルだけを事前学習済みResNetにする

::conc::
今回は「自分で一から学習」ではなく「学習済みの知識を借りる」

---

::title::
転移学習とは

::default::
**転移学習 (Transfer Learning)**

- 大量のデータで学習済みのモデルを、別のタスクに再利用する方法

- 今回は ImageNet で学習済みのResNetを、犬猫分類に使う

- 画像の基本的な特徴抽出は再利用し、最後の分類部分だけを犬猫用に変える

<br>

**なぜ有効か**

- 犬猫データが少なくても、すでに学習済みの画像特徴を使える

- SimpleCNNを最初から学習するより、少ないデータで高精度になりやすい

- 学習時間も短くしやすい

---
layout: two-cols
---

::title::
ImageNetで学習済みの特徴

::left::
<span class="font-bold">ImageNet</span>

- 多数の一般物体画像を含む大規模データセット

- 犬や猫だけでなく、車、鳥、道具、食べ物なども含む

- ResNetはこのデータから画像の特徴を学習済み

::right::
<span class="font-bold">再利用できる特徴</span>

- エッジ
- 色や模様
- 目・耳・輪郭のような部品
- 物体らしい形

<br>

犬猫分類でも、これらの特徴は役に立つ

::conc::
「犬猫専用の知識」ではなく「画像を見るための一般的な知識」を借りる

---
layout: two-cols
---

::title::
ResNet18の構造

::left::
<span class="font-bold">ResNet18の流れ</span>

```text
Input image
  ↓
Conv / BN / ReLU / Pool
  ↓
Residual Blocks
  ↓
Global Average Pooling
  ↓
fc: 1000 classes
```

::right::
<span class="font-bold">今回変更する場所</span>

```text
fc: 1000 classes
        ↓
fc: 2 classes
```

<br>

ImageNet用の1000クラス分類を、犬猫の2クラス分類に置き換える

::conc::
特徴抽出器は残し、最後の分類器だけを作り直す

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
<span class="font-bold">第六回の前処理</span>

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
- smallデータでも比較的安定しやすい
- まず授業ではこの形で実装する

::conc::
最初は「特徴抽出器を固定し、分類器だけ学習」が扱いやすい

---
layout: two-cols
---

::title::
精度をさらに上げたい場合

::left::
<span class="font-bold">Fine-tuning</span>

```python
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-4},
])
```

::right::
<span class="font-bold">注意点</span>

- 学習できる層を増やすと精度が上がることがある
- ただし、過学習もしやすくなる
- 学習率は小さめにする
- まずは `fc` だけで比較する

::conc::
Fine-tuningは発展として扱い、最初から全部を更新しない

---

::title::
学習コードへの組み込み

::default::
<Transform :scale="0.88">

```python
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

</Transform>

<br>

この後の学習ループは第六回とほぼ同じ

<br>

特徴抽出器を固定するため、学習フェーズでは `model.eval()` のあとに `model.fc.train()` を呼ぶ

---

::title::
SimpleCNNと比較する

::default::
**同じ条件で比較する**

- データ分割を変えない

- `batch_size` と `num_epochs` をそろえる

- Train Loss / Validation Loss / Validation Accuracy を同じグラフで見る

<br>

**見るべきポイント**

- smallデータでもValidation Accuracyが上がるか

- 学習時間が短くなっているか

- Train Lossだけ下がり、Validation Lossが悪化していないか

---

::title::
学習してみよう

::default::
- 第六回のColabノートブックをコピーして使う

- `SimpleCNN` を `models.resnet18(weights=weights)` に置き換える

- `model.fc` を2クラス分類用に差し替える

- まずは特徴抽出器を固定し、`model.fc` だけ学習する

- `data_size="small"`、`num_epochs=10` で実行して第六回と比較する

<br>

<span class="text-xl font-bold">Colabで `07_transfer_learning.ipynb` を実装してみましょう</span>
