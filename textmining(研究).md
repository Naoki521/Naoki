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

# ChatGPT×テキストマイニング(本体)

#APIkeyを入力
from getpass import getpass
secret = getpass('APIkeyを入力:')

import openai
openai.api_key = secret

# 実行コードになります。
import MeCab
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx
import collections
from textblob import TextBlob
import japanize_matplotlib
import requests
import sys

# DeepL翻訳関数
def translate_with_deepl(text, target_language, api_key):
    url = "https://api-free.deepl.com/v2/translate"
    data = {
        "auth_key": api_key,
        "text": text,
        "target_lang": target_language
    }
    response = requests.post(url, data=data)
    return response.json()["translations"][0]["text"]

# DeepL APIキー
deepl_api_key = "23d47694-30d9-ff97-72ef-aee13a565050:fx"

# レスポンスを格納するためのリスト
responses = []
Kresponses = []

#n回GPTに質問するかの入力(試験的に)
for i in range(1, 4):
    n = input("How many?(only integers):")
    try:
        val = int(n)
        break
    except ValueError:
        print("「整数いれてください。」")
    if i == 3:
        print("「整数が正しく入力されませんでした」")
        sys.exit()

# ユーザーからn回の入力を受け取る
for i in range(val):
    user_input = input(f"質問を入力({i+1}/{val}): ")

    #質問文を英語に翻訳
    user_input = translate_with_deepl(user_input, "EN", deepl_api_key)
    #print(user_input)

    # OpenAI GPT-3.5にリクエストを送信
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{user_input} within 100 tokens "}
        ]
    )

    # レスポンスをリストに追加
    responses.append(response["choices"][0]["message"]["content"])

    # GPT-3.5からのレスポンスを取得
    gpt_response = response["choices"][0]["message"]["content"]

    # レスポンスを日本語に翻訳
    translated_response = translate_with_deepl(gpt_response, "JA", deepl_api_key)

    # 翻訳されたレスポンスをリストに追加
    Kresponses.append(translated_response)

# 結果を表示
for i, res in enumerate(Kresponses, 1):
    print(f"「GPT」( {i}つ目): {res}")

# 形態素解析の準備
mecab = MeCab.Tagger('-Ochasen')

# 共起ネットワークの準備
co_occurrence_network = nx.Graph()

# 単語とセンチメントスコアのためのリスト
all_words = []
sentiments = []

#形態素解析実行
for response_text in Kresponses:
    node = mecab.parseToNode(response_text)
    words = []
    while node:
        word = node.surface
        pos = node.feature.split(",")[0]
        if pos not in ["助動詞", "助詞", "接続詞", "記号"]:
            words.append(word)
            all_words.append(word)
        node = node.next

# 単語頻度の計算
word_count = collections.Counter(all_words)

# 日本語フォントパスを設定
font_path = '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf'

# ワードクラウドの設定
wordcloud = WordCloud(font_path=font_path, background_color="white", width=800, height=400).generate_from_frequencies(word_count)

# ワードクラウドの表示
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# 共起ネットワークの準備
co_occurrence_network = nx.Graph()

#共起ネットワークの構築
for response_text2 in Kresponses:
    node = mecab.parseToNode(response_text2)
    words = []
    while node:
        word = node.surface
        pos = node.feature.split(",")[0]
        if pos in ["固有名詞", "動詞", "形容詞"]:
            words.append(word)
        node = node.next

    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            word1, word2 = words[i], words[j]
            if word1 != word2:
                if not co_occurrence_network.has_edge(word1, word2):
                    co_occurrence_network.add_edge(word1, word2, weight=1)
                else:
                    co_occurrence_network[word1][word2]['weight'] += 1

# 共起ネットワークの表示
#見づらいので、今後の課題
plt.figure(figsize=(12,12))
pos = nx.spring_layout(co_occurrence_network, k=0.5, iterations=50)  # ノードの位置をspring layoutで決定
nx.draw(co_occurrence_network, pos, with_labels=True, font_family= 'IPAexGothic', font_size=15)
plt.show()

# センチメント分析
for responses in responses:
    blob = TextBlob(responses)
    sentiments.append(blob.sentiment.polarity)

# センチメント分析の結果を可視化
plt.bar(range(len(sentiments)), sentiments)
plt.xlabel('GPT Index')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Analysis of Text')
plt.show()
