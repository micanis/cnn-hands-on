---
layout: two-cols
---

::title::
プログラムの大枠を設計する

::left::
```python
# Dataset / DataLoader
class CDDataset(Dataset):
  def __getitem__(self, idx):
    # 画像とラベルを返す

get_cd_dataloaders(
  root_path, data_size="small", batch_size=32
)

```

```python
# CNNの設計
class SimpleCNN(nn.Module):
  def __init__(self):
    # Conv -> Conv -> Linear -> Linear
  def forward(self, x):
    # ReLU / Pool / Flatten
```
::right::
```python
# 学習の前準備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
train_loss_list, val_loss_list, val_acc_list = [], [], []
```

```python
# 学習ループの実装
for epoch in range(num_epochs):
  # Train
  # Validation
  # Best Model Save
```

```python
# 推論・評価の実装
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
# 誤差の推移
# 正答率の推移

plt.show()
```
---

::title::
前処理の実装

::default::
**前処理とは**

- ネットワークモデルが効率よく学習できるよう、生データを適切な形に加工・管理する仕組みのこと

- `CDDataset`で画像ファイルを読み込み、ラベルとセットで返す

- `data_size`は`small: 20%`、`medium: 50%`、`large: 100%`を選べる

- [公式ドキュメント参照](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)

<br>

<span class="text-xl font-bold">Colabで `06_cnn.ipynb` を実装してみましょう</span>

---

::title::
ネットワークモデルの実装

::default::
**2層の畳み込み + 2層の全結合で実装**

- `nn.Module`を親クラスとして実装

- `forward`はモデル呼び出し時に自動で実行される

- 入力画像は`3 x 128 x 128`

- プーリング2回で`128 → 64 → 32`に縮小し、`32 * 32 * 32`を全結合層へ渡す

- [公式ドキュメント参照](https://docs.pytorch.org/tutorials/beginner/examples_nn/polynomial_module.html)

---

::title::
学習の前準備

::default::
- **損失関数**

  2クラス分類でも、出力を2次元にして `nn.CrossEntropyLoss()` を使う

- **最適化手法**

  まずは扱いやすい `Adam` で重みを更新する

- **記録用リスト**

  学習曲線を描くために、`train_loss_list`、`val_loss_list`、`val_acc_list`へ保存する

- **最良モデルの保存**

  検証損失が最も小さいモデルを`state_dict()`で保存する

---

::title::
学習ループの実装

::default::
- **学習フェーズ**

  `model.train()` に切り替え、バッチごとにGPUへ転送して学習する

- **検証フェーズ**

  `model.eval()` と `torch.no_grad()` で、重みを更新せずに損失と正答数を計算する

- **エポックごとの記録**

  平均損失と検証精度を表示し、検証損失が改善したらモデルを保存する

---

::title::
推論・評価の実装

::default::
- **推論**

  モデルの出力から最大値のクラスを選び、犬・猫の予測ラベルに変換する

- **精度評価**

  正解数をデータ数で割り、パーセントの正答率として確認する

- **学習曲線**

  `Train Loss`、`Validation Loss`、`Validation Accuracy`をグラフで確認する

---

::title::
学習してみよう

::default::

- **GPU**を使うこと

- `get_cd_dataloaders(..., data_size="small", batch_size=32)`で実行

- まずは授業時間内で回せるエポック数にする

- 損失が下がっているか、検証精度が上がっているかを確認する
