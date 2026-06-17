::title::
本日のまとめ

::default::
- **領域分割**
  
  ピクセルごとにクラスを決め、物体の輪郭まで表現するタスク

- **マスク**

  画像と同じ大きさの正解画像で、背景や各クラスを塗り分ける

- **モデルの種類**

  `DeepLabV3` などは semantic segmentation、`Mask R-CNN` は instance segmentation に使われる

- **推論**

  事前学習済みモデルに画像を入れ、`argmax` でマスクを作って重ね描画する
