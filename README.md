# 機械学習アルゴリズム
古典的なパターン認識や統計的機械学習、強化学習、深層学習など、様々な機械学習手法について勉強して、実装した内容を置いておくためのリポジトリです。

## 環境構築
GPU のある環境では ```--profile``` オプションに ```gpu``` を、CPU オンリーの環境では ```cpu``` を指定してビルドとコンテナ化を行います。

```bash
$ docker compose --profile <cpu/gpu> build
$ docker compose --profile <cpu/gpu> up -d
```

Jupyter Lab が立ち上がるので、[http://localhost:8888](http://localhost:8888) にアクセスします。

### GPU ありの場合
WSL2 上で NVIDIA のコンシューマ向け GPU を使って学習等を行うことを想定しています。
ベースとなる環境の構築は以下のサイトを参照してください。

- WSL2 のセットアップ：[WSL を使用して Windows に Linux をインストールする方法](https://learn.microsoft.com/ja-jp/windows/wsl/install)
- NVIDIA GPU Driver のインストール：[最新のドライバーのダウンロード](https://www.nvidia.com/ja-jp/drivers/)
- Docker のインストール：[Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

上記のような環境が整っている前提で、まずは [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) をインストールします。

```bash
# 必要なパッケージをインストール
$ sudo apt update
$ sudo apt install -y --no-install-recommends curl gnupg2

# リポジトリの追加とインストール
$ curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
$ sudo apt update
$ export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.0-1
$ sudo apt install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

Docker イメージは、[PyTorch 2.9.1 + CUDA 13.0 のイメージ](https://hub.docker.com/layers/pytorch/pytorch/2.9.1-cuda13.0-cudnn9-devel/images/sha256-d8d98ebdc7006e495d263d8734eb7bb2d19803419d9315159fe15c62d5bad1bd)をベースに作成しています。

### GPU なしの場合
GPU ありの環境でベースにしている PyTorch の公式イメージの環境に可能な限り近づけています。
ただし、Conda や NVIDIA 関連のパッケージはインストールしていません。

## 参考書
###  ケヴィン P. マーフィー 著, 持橋大地・鈴木大慈 監訳 『[確率的機械学習：入門編 I](https://www.asakura.co.jp/detail.php?book_code=12303)・[II](https://www.asakura.co.jp/detail.php?book_code=12304)』(朝倉書店, 2025)
- つい最近出された (2025年11月発売)、機械学習の理論をまとめた2分冊の訳書です。 
- 古典的な機械学習のトピックに加え深層学習についても取り扱っており、内容的に下の２冊からアップデートされています。
- 原著: Kevin P. Murphy “[Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html)“ (MIT Press, 2022)

### Christopher M. Bishop "[Patern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)" (Springer, 2006)
- リンク先は無料で入手可能な英語版です。
- PRML 本とも呼ばれる機械学習界隈で知らぬ人のいない大著で、実装メインのハウツー本というよりは、理論をしっかり理解するための本です。
- 日本語版：元田浩・栗田多喜夫・樋口知之・松本裕治・村田昇 監訳『[パターン認識と機械学習 上](https://www.maruzen-publishing.co.jp/book/b10111651.html)/[下](https://www.maruzen-publishing.co.jp/book/b10111678.html)』(丸善出版, 2012)

### Trevor Hastie, Robert Tibshirani, and Jerome Friedman "[The Elements of Statistical Learning](https://www.sas.upenn.edu/~fdiebold/NoHesitations/BookAdvanced.pdf)" (Springer, 2009)
- PRML 本と同様、リンク先は無料で入手可能な英語版です。
- 別名「カステラ本」で、PRML 本と並ぶ機械学習の名著です。こちらもかなり理論寄りの内容になります。
- 日本語版： 杉山将・井手剛・神嶌敏弘・栗田多喜夫・前田英作 監訳『[統計的学習の基礎](https://www.kyoritsu-pub.co.jp/book/b10004471.html)』(共立出版, 2014)

### Richard S. Sutton・Andrew G. Barto 著, 奥村エルネスト純・鈴木雅大・松尾豊・三上貞芳・山川宏 監訳 『[強化学習 第2版](https://www.morikita.co.jp/books/mid/082662)』 (森北出版, 2022)
- 強化学習の理論全般に加え、心理学や神経科学との関連や AlphaGo についてもまとめられた書籍の訳本です。
- 数式による理論の説明と、疑似コードによるアルゴリズムの解説が中心で、実装にはほぼ触れられていません。
- 原著: Richard S. Sutton and Andrew G. Barto "[Reinforcement Learning: An Introduction Second Edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)" (Bradford Books, 2018)

### Ian Goodfellow・Yoshua Bengio・Aaron Courville 著, 岩澤有祐・鈴木雅大・中山浩太郎・松尾豊 監訳 『[深層学習](https://asciidwango.jp/post/171302668055/深層学習)』　(アスキードワンゴ, 2018)
- 深層学習の生みの親が執筆に携わった、言わずと知れた名著です。
- CNN や RNN の理論がまとめられていますが、Transformer などの登場前の本なので、近年の動向まではカバーしきれていません。
- 原著: Ian Goodfellow, Yoshua Bengio, and Aaron Courville "[Deep Learning](https://www.deeplearningbook.org)" (MIT Press, 2016)