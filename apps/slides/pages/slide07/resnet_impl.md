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
    data_size="large",
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

# 第七回 前半
model = ResNet18(num_classes=2).to(device)
```

<br>

- Dataset / DataLoaderは同じ
- 学習ループも同じ
- モデルだけを自作ResNet18にする

::conc::
まずは自分でResNetを実装し、SimpleCNNとの違いを確認する

---
layout: two-cols
---

::title::
BasicBlockの実装

::left::
<Transform :scale="0.74">

```python
class BasicBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    super().__init__()
    self.conv1 = nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3, stride=stride,
        padding=1, bias=False
    )
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(
        out_channels, out_channels,
        kernel_size=3, padding=1,
        bias=False
    )
    self.bn2 = nn.BatchNorm2d(out_channels)
```

</Transform>

::right::
<Transform :scale="0.74">

```python
    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
          nn.Conv2d(
              in_channels, out_channels,
              kernel_size=1,
              stride=stride,
              bias=False
          ),
          nn.BatchNorm2d(out_channels),
      )
```

</Transform>

<br>

- `stride=2` のとき画像サイズを半分にする
- チャンネル数が変わるときは `1x1 Conv` で形を合わせる

---

::title::
BasicBlockのforward

::default::
<Transform :scale="0.9">

```python
def forward(self, x):
  identity = self.shortcut(x)

  out = self.conv1(x)
  out = self.bn1(out)
  out = F.relu(out)

  out = self.conv2(out)
  out = self.bn2(out)

  out = out + identity
  out = F.relu(out)
  return out
```

</Transform>

<br>

- `identity` は足し戻す経路
- `out + identity` で残差接続を作る
- 足し算するため、`out` と `identity` の形をそろえる

---
layout: two-cols
---

::title::
ResNet18の全体構造

::left::
<span class="font-bold">入力サイズの変化</span>

```text
Input:   3 x 128 x 128
Stem:   64 x 32 x 32
Layer1: 64 x 32 x 32
Layer2: 128 x 16 x 16
Layer3: 256 x 8 x 8
Layer4: 512 x 4 x 4
GAP:    512 x 1 x 1
Linear: 2
```

::right::
<span class="font-bold">実装する部品</span>

- `stem`
- `layer1`
- `layer2`
- `layer3`
- `layer4`
- `avgpool`
- `fc`

<br>

最後は犬猫の2クラス分類なので、`fc` の出力は2

::conc::
空間サイズを小さくしながら、チャンネル数を増やして特徴を豊かにする

---
layout: two-cols
---

::title::
ResNet18クラスの実装

::left::
<Transform :scale="0.7">

```python
class ResNet18(nn.Module):
  def __init__(self, num_classes=2):
    super().__init__()
    self.in_channels = 64

    self.stem = nn.Sequential(
        nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2, padding=1),
    )

    self.layer1 = self._make_layer(64, blocks=2, stride=1)
    self.layer2 = self._make_layer(128, blocks=2, stride=2)
    self.layer3 = self._make_layer(256, blocks=2, stride=2)
    self.layer4 = self._make_layer(512, blocks=2, stride=2)
```

</Transform>

::right::
<Transform :scale="0.7">

```python
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, num_classes)

  def _make_layer(self, out_channels, blocks, stride):
    layers = [BasicBlock(self.in_channels, out_channels, stride)]
    self.in_channels = out_channels

    for _ in range(1, blocks):
      layers.append(BasicBlock(out_channels, out_channels))

    return nn.Sequential(*layers)
```

</Transform>

---

::title::
ResNet18のforward

::default::
<Transform :scale="0.95">

```python
def forward(self, x):
  x = self.stem(x)
  x = self.layer1(x)
  x = self.layer2(x)
  x = self.layer3(x)
  x = self.layer4(x)
  x = self.avgpool(x)
  x = torch.flatten(x, 1)
  x = self.fc(x)
  return x

model = ResNet18(num_classes=2).to(device)
```

</Transform>

<br>

`CrossEntropyLoss` を使うため、モデルの最後にSoftmaxは入れない

---

::title::
自作ResNetを学習してみよう

::default::
- `07_resnet.ipynb` を開く

- `SimpleCNN` を自作 `ResNet18` に差し替える

- まずは `data_size="large"`、`num_epochs=5` で実行する

- Validation Accuracyがどこまで上がるか確認する

- 目安として、85-90%前後を狙う

<br>

<span class="text-xl font-bold">Colabで `07_resnet.ipynb` を実装してみましょう</span>
