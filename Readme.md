# 关键词提取
关键词提取 (Keyphrase Extraction，KPE) 任务可以自动提取文档中能够概括核心内容的短语，有利于下游信息检索和 NLP 任务。当前，由于对文档进行标注需要耗费大量资源且缺乏大规模的关键词提取数据集，无监督的关键词提取在实际应用中更为广泛。无监督关键词抽取的state of the art（SOTA）方法是对候选词和文档标识之间的相似度进行排序来选择关键词。但由于候选词和文档序列长度之间的差异导致了关键短语候选和文档的表征不匹配，导致以往的方法在长文档上的性能不佳，无法充分利用预训练模型的上下文信息对短语构建表征。
## 数据集
  数据集来自知网和维普网，知网、维普网等数据集的关键词提取可以作为知识发现的一种途径，由于当前关键词提取的各类算法各有优劣，基于统计学的算法依赖切分词效果且缺失上下文语义信息，基于预训练模型的算法偏向于获取长度较长的短语，且在英文数据集效果好于中文数据集，可以尝试结合各类算法结果互补，在缺乏专家知识情况下得到较优的新词发现结果，然后获得细粒度的切分词效果，然后基于词信息熵约束构建整个概念权重网络。
## 基于词袋加权的TFIDF算法
  TF-IDF是一种统计方法，用以评估一个字词对于一个文件集或一个语料库中的其中一份文件的重要程度，字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降，也就是说一个词语在一篇文章中出现次数越多, 同时在所有文档中出现次数越少, 越能够代表该文章。如果词w在一篇文档d中出现的频率高，并且在其他文档中很少出现，则认为词w具有很好的区分能力，适合用来把文章d和其他文章区分开来。
  ```python
import jieba.analyse as analyse

text = '''注重数据整合，……风险防控平台。'''

jieba.analyse.extract_tags(text, topK=20, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v','nr', 'nt'))
  ```

 TF-IDF算法的优点是简单快速，结果比较符合实际情况。缺点是，**单纯以"词频"衡量一个词的重要性，不够全面，有时重要的词可能出现次数并不多。**

**此外，这种算法无法体现词的位置信息，出现位置靠前的词与出现位置靠后的词，都被视为重要性相同，IDF的简单结构并不能有效地反映单词的重要程度和特征词的分布情况，使其无法很好地完成对权值调整。**
## 考虑词关联网络的TextRank算法
上面说到，TF-IDF基于词袋模型（Bag-of-Words），把文章表示成词汇的集合，由于集合中词汇元素之间的顺序位置与集合内容无关，所以TF-IDF指标不能有效反映文章内部的词汇组织结构。

TextRank由Mihalcea与Tarau提出，通过词之间的相邻关系构建网络，然后用PageRank迭代计算每个节点的rank值，排序rank值即可得到关键词。

TextRank是一种基于随机游走的关键词提取算法，考虑到不同词对可能有不同的共现（co-occurrence），TextRank将共现作为无向图边的权值。
其实现包括以下步骤：
（1）把给定的文本T按照完整句子进行分割；
（2）对于每个句子，进行分词和词性标注处理，并过滤掉停用词，只保留指定词性的单词，如名词、动词、形容词，即，其中 ti,j 是保留后的候选关键词；
（3）构建候选关键词图G = (V,E)，其中V为节点集，由2)生成的候选关键词组成，然后采用共现关系（co-occurrence）构造任两点之间的边，两个节点之间存在边仅当它们对应的词汇在长度为K的窗口中共现，K表示窗口大小，即最多共现K个单词；
（4）根据上面公式，迭代传播各节点的权重，直至收敛；
（5）对节点权重进行倒序排序，从而得到最重要的T个单词，作为候选关键词；
（6）由5得到最重要的T个单词，在原始文本中进行标记，若形成相邻词组，则组合成多词关键词；
```python
text = '''注重数据整合，……风险防控平台。'''

jieba.analyse.textrank(text, topK=20, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v','nr', 'nt'))
```
TextRank与TFIDF均严重依赖于分词结果—如果某词在分词时被切分成了两个词，那么在做关键词提取时无法将两个词黏合在一起

不过，TextRank虽然考虑到了词之间的关系，但是仍然倾向于将频繁词作为关键词。
## 结合主题的LDA算法

LDA（Latent Dirichlet Allocation）是一种文档主题生成模型，也称为一个三层贝叶斯概率模型，包含词、主题和文档三层结构。

所谓生成模型，就是说，我们认为一篇文章的每个词都是通过“以一定概率选择了某个主题，并从这个主题中以一定概率选择某个词语”这样一个过程得到。文档到主题服从多项式分布，主题到词服从多项式分布。

因此计算词的分布和文档的分布的相似度，取相似度最高的几个词作为关键词。

```python
lda_news = hub.Module(name="lda_news")

lda_news.cal_doc_keywords_similarity(text)
```
LDA通过主题建模，在一定程度上考虑了文档与关键词在主题上的一致性，但这需要找到合适的主题数目作为超参数，具体的效果受其影响较大。

## 结合语义编码的KeyBert算法
KeyBERT(Sharma, P., & Li, Y. (2019). Self-Supervised Contextual Keyword and Keyphrase Retrieval with Self-Labelling)，提出了一个利用bert快速提取关键词的方法。

原理十分简单：首先使用 BERT 提取文档嵌入以获得文档级向量表示。随后，为N-gram 词/短语提取词向量，然后，我们使用余弦相似度来找到与文档最相似的单词/短语。最后可以将最相似的词识别为最能描述整个文档的词。

```python
from keybert import KeyBERT
kw_model = KeyBERT(model='paraphrase-MiniLM-L6-v2')
keywords = kw_model.extract_keywords(text)
print("\nkeyphrase_ngram_range ...")
keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words=None)
keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), highlight=None)
# 为了使结果多样化，我们将 2 x top_n 与文档最相似的词/短语。
# 然后，我们从 2 x top_n 单词中取出所有 top_n 组合，并通过余弦相似度提取彼此最不相似的组合。
print("\nuse_maxsum ...")
keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words=None,
                              use_maxsum=True, nr_candidates=20, top_n=5)
# 为了使结果多样化，我们可以使用最大边界相关算法(MMR)

# 来创建同样基于余弦相似度的关键字/关键短语。 具有高度多样性的结果：

print("\nhight diversity ...")
keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(3, 3), stop_words=None,use_mmr=True, diversity=0.7)
print("\nlow diversity ...")
keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(3, 3), stop_words=None, use_mmr=True, diversity=0.2)
```



Keybert基于一种假设，关键词与文档在语义表示上是一致的，利用bert的编码能力，能够得到较好的结果。

但缺点很明显：

首先，不同的语义编码模型会产生不同的结果，这个比较重要。

此外，由于bert只能接受限定长度的文本，例如512个字，这个使得我们在处理长文本时，需要进一步加入摘要提取等预处理措施，这无疑会带来精度损失。

## Yake
  它是一种轻量级、无监督的自动关键词提取方法，它依赖于从单个文档中提取的统计文本特征来识别文本中最相关的关键词。该方法不需要针对特定的文档集进行训练，也不依赖于字典、文本大小、领域或语言。Yake 定义了一组五个特征来捕捉关键词特征，这些特征被启发式地组合起来，为每个关键词分配一个分数。分数越低，关键字越重要。

特征提取主要考虑五个因素(去除停用词后)
大写字母的term（除了每句话的开头单词）的重要程度比那些小写字母的term重要程度要大。
词的位置， 文本越开头的部分句子的重要程度比后面的句子重要程度要大。
词频， 一个词在文本中出现的频率越大，相对来说越重要，同时为了避免长文本词频越高的问题，会进行归一化操作。
上下文关系， 一个词与越多不相同的词共现，该词的重要程度越低。
```python
full_text = title +", "+ text

full_text = " ".join(jieba.cut(full_text))

kw_extractor = yake.KeywordExtractor(top=10, n=1,stopwords=None)

keywords = kw_extractor.extract_keywords(full_text)

 print("Keyphrase: ",kw, ": score", v)
```
## Rake
Rake 是 Rapid Automatic Keyword Extraction 的缩写，它是一种从单个文档中提取关键字的方法。实际上提取的是关键的短语(phrase)，并且倾向于较长的短语，在英文中，关键词通常包括多个单词，但很少包含标点符号和停用词，例如and，the，of等，以及其他不包含语义信息的单词。

Rake算法首先使用标点符号（如半角的句号、问号、感叹号、逗号等）将一篇文档分成若干分句，然后对于每一个分句，使用停用词作为分隔符将分句分为若干短语，这些短语作为最终提取出的关键词的候选词。

每个短语可以再通过空格分为若干个单词，可以通过给每个单词赋予一个得分，通过累加得到每个短语的得分。Rake 通过分析单词的出现及其与文本中其他单词的兼容性（共现）来识别文本中的关键短语。最终定义的公式是:

$$
wordScore = wordDegree(w)/wordFrequency(w)
$$

即单词的得分是该单词的度（是一个网络中的概念，每与一个单词共现在一个短语中，度就加1，考虑该单词本身）除以该单词的词频（该单词在该文档中出现的总次数）。

然后对于每个候选的关键短语，将其中每个单词的得分累加，并进行排序，RAKE将候选短语总数的前三分之一的认为是抽取出的关键词。
```python
import jieba.posseg as pseg

from collections import Counter

# Data structure for holding data

    def __init__(self, char, freq = 0, deg = 0):

    def returnScore(self):

        return self.deg/self.freq

    def updateOccur(self, phraseLength):

        self.deg += phraseLength

    def updateFreq(self):

        if '\u0041' <= item <= '\u005a' or ('\u0061' <= item <='\u007a') or item.isdigit():

# Read Target Case if Json

def readSingleTestCases(testFile):

    with open(testFile) as json_data:

            testData = json.load(json_data)

            # This try block deals with incorrect json format that has ' instead of "

            data = json_data.read().replace("'",'"')

                testData = json.loads(data)

                # This try block deals with empty transcript file

                return ""

    for item in testData:

            returnString += item['text']

            returnString += item['statement']

    # Construct Stopword Lib

    swLibList = [line.rstrip('\n') for line in open(r"../../../stopword/stopwords.txt",'r',encoding='utf-8')]

    # Construct Phrase Deliminator Lib

    conjLibList = [line.rstrip('\n') for line in open(r"wiki_quality.txt",'r',encoding='utf-8')]

    rawtextList = pseg.cut(rawText)

    # Construct List of Phrases and Preliminary textList

    listofSingleWord = dict()

    poSPrty = ['m','x','uj','ul','mq','u','v','f']

    for eachWord, flag in rawtextList:

        checklist.append([eachWord,flag])

        if eachWord in conjLibList or not notNumStr(eachWord) or eachWord in swLibList or flag in poSPrty or eachWord == '\n':

            if lastWord != '|':

                textList.append("|")

                lastWord = "|"

        elif eachWord not in swLibList and eachWord != '\n':

            textList.append(eachWord)

            meaningfulCount += 1

            if eachWord not in listofSingleWord:

                listofSingleWord[eachWord] = Word(eachWord)

            lastWord = ''

    # Construct List of list that has phrases as wrds

    for everyWord in textList:

        if everyWord != '|':

            tempList.append(everyWord)

            newList.append(tempList)

            tempList = []

    for everyWord in textList:

        if everyWord != '|':

            tempStr += everyWord + '|'

            if tempStr[:-1] not in listofSingleWord:

                listofSingleWord[tempStr[:-1]] = Word(tempStr[:-1])

                tempStr = ''

    # Update the entire List

    for everyPhrase in newList:

        for everyWord in everyPhrase:

            listofSingleWord[everyWord].updateOccur(len(everyPhrase))

            res += everyWord + '|'

        phraseKey = res[:-1]

        if phraseKey not in listofSingleWord:

            listofSingleWord[phraseKey] = Word(phraseKey)

            listofSingleWord[phraseKey].updateFreq()

    # Get score for entire Set

    for everyPhrase in newList:

        if len(everyPhrase) > 5:

        phraseString = ''

        for everyWord in everyPhrase:

            score += listofSingleWord[everyWord].returnScore()

            phraseString += everyWord + '|'

            outStr += everyWord

        phraseKey = phraseString[:-1]

        freq = listofSingleWord[phraseKey].getFreq()

        if freq / meaningfulCount < 0.01 and freq < 3 :

        outputList[outStr] = score

    sorted_list = sorted(outputList.items(), key = operator.itemgetter(1), reverse = True)

    return sorted_list[:50]

if __name__ == '__main__':

    with open(r'Ai_zhaiyaohebing.txt','r') as fp:

        for i in range(100):

            text += fp.readline()

        result = run(text)
```

## Autophrasex
Autophrasex是新词发现算法，主要思想是：

Robust Positive-Only Distant Training：使用wiki和freebase作为先验数据，根据知识库中的相关数据构建Positive Phrases,根据领域内的文本生成Negative Phrases，构建分类器后根据预测的结果减少负标签带来的噪音问题。

POS-Guided Phrasal Segmentation：使用POS词性标注的结果，引导短语分词，利用POS的浅层句法分析的结果优化Phrase

AutoPhrase可以支持任何语言，只要该语言中有通用知识库。与当下最先进的方法比较，新方法在跨不同领域和语言的5个实际数据集上的有效性有了显著提高。

```python
pip install autophrasex

from autophrasex import *

    reader=DefaultCorpusReader(tokenizer=JiebaTokenizer()),

    selector=DefaultPhraseSelector(),

        NgramsExtractor(N=4), 

        EntropyExtractor()

predictions = autophrase.mine(

    corpus_files=['./Aizhaiyao.txt'],

    quality_phrase_files='./wiki_quality.txt',

        LoggingCallback(),

        ConstantThresholdScheduler(),

        EarlyStopping(patience=2, min_delta=5)
```
## MDERank
https://github.com/LinhanZ/mderank
1.由原先的短语-文本层面的相似度计算通过掩码转换为文本-文本层面，从而解决了长度不匹配的问题

2.避免了使用短语的表征，而是使用掩码后的文本替代候选词，由此可以获得充分利用了上下文信息的文本表征

![](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231227153044599.png)



**1.绝对采样：利用现有的无监督关键词提取方法，对每篇文章抽取得到关键词集合，将这些“关键词”作为正例对原始文章进行掩码。然后在剩下候选词（“非关键词”）中随机抽取作为负例。**

2.相对采样：利用现有的无监督关键词提取方法，对每篇文章抽取得到 d 关键词集合，在关键词集合中，随机抽取两个“关键词”，其中排名靠前的一个作为正例，另一个则为负例，从而构建训练数据。

![](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231227153148301.png)
## 其他开源代码
https://github.com/sunyilgdx/SIFRank_zh
https://github.com/fighting41love/funNLP