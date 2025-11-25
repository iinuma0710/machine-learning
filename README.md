# 機械学習アルゴリズム
古典的なパターン認識や統計的機械学習、強化学習、深層学習など、様々な機械学習手法について勉強して、実装した内容を置いておくためのリポジトリです。

## 環境構築
### GPU なしの場合


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

