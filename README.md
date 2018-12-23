# autokeras を使った文書分類器

[Auto-Keras](https://autokeras.com/) を用いた日本語対応の文書分類器。

## セットアップ

1. GPU 環境を用意 (用意できない場合、 `requirements.yaml` を適宜書き換えてください)
2. Anaconda 3 をインストール
3. `$ conda env create --file requirements.yaml`

## 使い方

各スクリプトは `-h` オプションを渡すことでヘルプを参照できる。

```bash
$ ./build-dataset.py -h
```

### データセットの作成

`build-dataset.py` を使う。
1 行 1 文章なテキストファイルを、ポジティブ・ネガティブのそれぞれに用意し、さらにその 2 つを結合したデータを用意することになる。

### 学習

`train.py` を使う。

### 判定

`simple_predictor.py` を使う。
対話環境で文書の判定ができるが、最適化されていないので実行には時間がかかる。
