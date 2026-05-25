---
layout: two-cols
---

::title::
MNISTデータセット

::left::
<span class="font-bold">MNISTとは?</span>

- <span class="font-bold">手書き数字 (0~9) </span>の画像データセット
- 訓練データ: 60,000枚
- テストデータ: 10,000枚
- 画像サイズ: **28×28 グレースケール**
- 10クラス分類タスク

::right::
<span class="font-bold">なぜMNISTか</span>

- 第4回で作ったCNNと**そのまま整合**
  - 入力: (1, 28, 28)、出力: 10クラス
- PyTorchから直接ダウンロード可能
- 学習時間が短い（GPUなしでもOK）


::conc::
まずはMNISTで学習の流れを学び、次回の犬猫分類に活かす

---
layout: two-cols
---

::title::
PyTorchでのデータ読み込み

::left::
<span class="font-bold">torchvision.datasets を使用</span>

<Transform :scale="0.65">

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 前処理: 画像をテンソルに変換
transform = transforms.Compose([
    transforms.ToTensor(),
])

# データのダウンロードと読み込み
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform,
)
```

</Transform>

::right::
<span class="font-bold">DataLoaderでバッチ化</span>

<Transform :scale="0.85">

```python
# バッチサイズ64でデータを取り出す
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
)

# 確認
images, labels = next(iter(train_loader))
print(images.shape)  # (64, 1, 28, 28)
print(labels.shape)  # (64,)
```

</Transform>

::conc::
DataLoaderがデータをバッチ単位で自動的に供給してくれる
