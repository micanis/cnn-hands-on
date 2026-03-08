import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第2回で学ぶ：畳み込み層
        # 入力3チャンネル(RGB) -> 出力16チャンネルへ特徴を抽出
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # 入力16チャンネル -> 出力32チャンネルへさらに深く抽出
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # 第4回で学ぶ：全結合層
        # 128x128の画像が2回のプーリング(1/2)で 32x32 になる
        # 32(チャンネル) * 32(縦) * 32(横) = 32768
        self.fc1 = nn.Linear(in_features=32 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2) # 最終出力は2クラス（犬/猫）

    def forward(self, x):
        # 第3回・第4回で学ぶ：データの流れ（順伝播）
        # 畳み込み -> ReLU(活性化) -> MaxPool(圧縮) のセット
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        
        # 1列に並べ直す（Flatten）
        x = torch.flatten(x, 1)
        
        # 全結合層
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x