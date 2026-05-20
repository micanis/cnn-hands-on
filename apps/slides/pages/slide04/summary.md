::title::
本日のまとめ

::default::

- **全結合層**
  
  特徴マップを1次元化(Flatten)し、分類を行う層
  
  $y = Wx + b$ の行列演算、すべての入力と出力が接続

- **ネットワークの構築**

  Conv → ReLU → Pool → ... → Flatten → Linear の流れ

  `nn.Sequential` で簡単に、`nn.Module` で柔軟に構築

- **次回予告**

  <span class="text-blue-500 font-bold">損失関数</span>と<span class="text-blue-500 font-bold">最適化</span>を学び、ネットワークを「学習」させる
