---
layout: two-cols
---

::title::
推論の流れ

::left::
<span class="font-bold">手順</span>

1. 画像を読み込む
2. 前処理を行う
3. モデルに入力する
4. ピクセルごとの予測を受け取る
5. マスクとして描画する

<br>

出力は、クラスごとのスコアが並んだ 3 次元テンソルになる

::right::
<span class="font-bold">イメージ</span>

```text
input:  [3, H, W]
output: [C, H, W]
mask:   [H, W]
```

<br>

`argmax` を取ると、各ピクセルの代表クラスが決まる

::conc::
モデルの出力をそのまま読むのではなく、マスクに変換して可視化する

---
layout: two-cols
---

::title::
推論コードの例

::left::
<Transform :scale="0.82">

```python
import torch
from PIL import Image
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights
)

weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights)
model.eval()
```

</Transform>

::right::
<Transform :scale="0.82">

```python
image = Image.open("sample.jpg").convert("RGB")
batch = weights.transforms()(image).unsqueeze(0)

with torch.no_grad():
    output = model(batch)["out"][0]

mask = output.argmax(0)
```

</Transform>

<br>

- `weights.transforms()` で前処理をそろえる
- `model.eval()` で推論モードにする
- `output["out"]` には各クラスのスコアが入る


