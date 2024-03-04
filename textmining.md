# ChatGPT×テキストマイニング(事前準備)
#開発環境はGoogle colaboratory
#naoki abe

#Mecabをインストール
!apt install aptitude
!aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file -y
!pip install mecab-python3==0.7

#matplotlibをインストール
!pip install japanize-matplotlib

#フォントパッケージのインストール
!apt-get -y install fonts-ipafont-gothic
!rm /root/.cache/matplotlib -rf
!fc-list :lang=ja

#ワードクラウド生成ツールをインストール
!pip install wordcloud

#共起ネットワーク生成用ツールインストール
!pip install networkx

#OpenAIをインストール
!pip install openai==0.28

#DeepLへのリクエスト送信のためのコード
!pip install requests
