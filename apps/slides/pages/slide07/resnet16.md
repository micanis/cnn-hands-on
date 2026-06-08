---
layout: two-cols
---

::title::
第六回コードからの変更点

::left::
<span class="font-bold">そのまま使う部分</span>

```python
train_loader, val_loader = get_cd_dataloaders(
    root_path,
    data_size="small",
    batch_size=32
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
model = ResNet16(num_classes=2).to(device)
```

<br>

- データセットの読み込みは同じ
- 損失関数も同じ
- 学習ループも同じ
- ネットワークだけを深くする

::conc::
まずは「動いている学習コード」の一部だけを安全に置き換える

---

::title::
なぜ深いネットワークにするのか

::default::
**SimpleCNNの限界**

- 層が浅いと、単純な特徴しか組み合わせにくい

- 犬猫分類では、耳・目・毛並み・輪郭・背景など複数の特徴を見る必要がある

- ただし、単純に層を増やすと学習が不安定になることがある

<br>

**ResNetの考え方**

- 畳み込みの結果だけでなく、入力そのものも次の層へ渡す

- 「入力からどれだけ変えればよいか」を学習する

- 深くしても勾配が伝わりやすくなる

---
layout: two-cols
---

::title::
残差接続のイメージ

::left::
<span class="font-bold">通常の畳み込みブロック</span>

```python
y = conv_block(x)
```

<br>

<div class="flex flex-col gap-1 items-center text-sm">
  <div class="px-4 py-1 bg-gray-100 border rounded">x</div>
  <span>↓</span>
  <div class="px-4 py-1 bg-blue-100 border rounded">Conv / BN / ReLU</div>
  <span>↓</span>
  <div class="px-4 py-1 bg-gray-100 border rounded">y</div>
</div>

::right::
<span class="font-bold">ResNetのBasicBlock</span>

```python
y = conv_block(x) + shortcut(x)
```

<br>

<div class="flex flex-col gap-1 items-center text-sm">
  <div class="px-4 py-1 bg-gray-100 border rounded">x</div>
  <span>↓</span>
  <div class="px-4 py-1 bg-blue-100 border rounded">Conv / BN / ReLU / Conv / BN</div>
  <span>+</span>
  <div class="px-4 py-1 bg-green-100 border rounded">shortcut(x)</div>
  <span>↓</span>
  <div class="px-4 py-1 bg-orange-100 border rounded">ReLU</div>
</div>

::conc::
出力は「変換した特徴」と「元の特徴」の足し算になる

---
layout: two-cols
---

::title::
BasicBlockの実装

::left::
<Transform :scale="0.78">

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
```

</Transform>

::right::
<Transform :scale="0.78">

```python
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
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
- 足し算するため、`out` と `identity` の形をそろえる必要がある

---
layout: two-cols
---

::title::
ResNet16の全体構造

::left::
<span class="font-bold">入力サイズ</span>

```text
Input:    3 x 128 x 128
Stem:    32 x 64 x 64
Pool:    32 x 32 x 32
Layer1:  32 x 32 x 32
Layer2:  64 x 16 x 16
Layer3: 128 x 8 x 8
GAP:     128 x 1 x 1
Linear:  2
```

::right::
<span class="font-bold">層の数え方</span>

- Stem Conv: 1層
- BasicBlock: 7個 × 2 Conv = 14層
- Linear: 1層

<br>

<span class="text-xl font-bold text-blue-500">合計 16層</span>

<br>

`[2, 2, 3]` 個のBasicBlockを3段に分けて積む

::conc::
第六回より深いが、授業内で追える小さめのResNetにする

---

::title::
動作確認

::default::
<Transform :scale="0.95">

```python
model = ResNet16(num_classes=2)

x = torch.randn(4, 3, 128, 128)
out = model(x)

print(out.shape)
```

</Transform>

<br>

```text
torch.Size([4, 2])
```

<br>

- バッチサイズ4の画像を入力
- 出力は犬・猫の2クラス分のスコア
- `CrossEntropyLoss` に渡すため、Softmaxはモデル内に入れない

---
layout: two-cols
---

::title::
学習コードへの組み込み

::left::
<span class="font-bold">モデルを作る</span>

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = ResNet16(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)
```

::right::
<span class="font-bold">第六回と同じ流れで学習</span>

```python
num_epochs = 10
train_loss_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    # train
    # validation
    # best model save
```

<br>

- `weight_decay` は重みが大きくなりすぎることを抑える
- まずは第六回と同じ `data_size="small"` で比較する

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

- Validation Accuracyが上がったか

- Train Lossだけ下がり、Validation Lossが悪化していないか

- 学習時間が増えすぎていないか

---

::title::
学習してみよう

::default::
- 第六回のColabノートブックをコピーして使う

- `SimpleCNN` の定義を `BasicBlock` と `ResNet16` に置き換える

- `model = ResNet16(num_classes=2).to(device)` に変更する

- まずは `data_size="small"`、`num_epochs=10` で実行する

- 精度が伸びる場合は `medium`、`large` でも試す

<br>

<span class="text-xl font-bold">Colabで `07_resnet16.ipynb` を実装してみましょう</span>
