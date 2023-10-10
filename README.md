# TokiPona Transformer
TokiPonaを使用して個人でもLLMを学習可能にした(しようとした)Transformer
## TokiPonaとは？
- 人工言語のひとつ
-言語全体の単語の数が120程しかないため、Transformerのone-hot encodingと相性が良い
## アイデア
- 英語でできたデータセットをTokiPonaに落とし込む
    - 翻訳モデルはHugging Faceからダウンロードされる
- TokiPonaになったデータセットでLLMを学習
    - 単語数が翻訳によって圧縮される・文の多様性が制限されるので元のデータセットに比べて非常に学習が容易になる
- 文を生成
