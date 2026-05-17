::header::
本日のまとめ

::default::

- **プーリング層**
  
  特徴マップを縮小し、位置ズレに強くする
  
  MaxPooling（最大値）が主流、学習パラメータなし


- **活性化関数**

  非線形性を加え、深いネットワークに意味を持たせる

  ReLU（負→0、正→そのまま）が主流、計算が単純で勾配消失に強い


- **CNNの基本ブロック**

  <span class="text-blue-500 font-bold">Conv → ReLU → Pool</span> の組み合わせが1セット

