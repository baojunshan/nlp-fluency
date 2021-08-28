# nlp-fluency

## 说明
- 评估自然语言的流畅度的方法集
- 包括`ngrams`, `gpt`, `masked bert`几种不同计算流畅度的方法；kenlm的方法可以参考苏神的[博客](https://spaces.ac.cn/archives/3956)
- 欢迎star，issue以及PR
  
|方法|介绍|模型|案例|
|:---|:---|:---|:---|
|ngrams|利用ngram计算得到下一个词的概率【单向滑窗】|[百度网盘:no8i](https://pan.baidu.com/s/16-4EhOIWCWxarjgGYh93bQ) （基于thucnew摘要数据集训练）； 也可以通过[train_ngramslm.py]()用其他语料训练|[案例](https://github.com/baojunshan/nlp-fluency#ngrams)|
|gpt|利用中文gpt计算得到下一个词的概率【单向】|[百度网盘:qmzg](https://pan.baidu.com/s/1R8BRDiLfW8jzhpB3adtiTg) ； 也可以访问 [链接](https://github.com/Morizeyao/GPT2-Chinese) 获取其他gpt预训练的中文模型，或者自己训练|[案例](https://github.com/baojunshan/nlp-fluency#gpt)|
|bert|把句子中的词mask住，然后预测得到mask词的分布，进而得到该词的概率【双向】|[百度网盘:ma3b](https://pan.baidu.com/s/18qMsM0wqL_r2j1qxohSDNA) ； 也可以访问[链接](https://github.com/ymcui/Chinese-BERT-wwm) 获取其他bert预训练的中文模型，或者自己训练|[案例](https://github.com/baojunshan/nlp-fluency#bert)|
|albert|同bert，只是模型小了|[百度网盘:q6pb](https://pan.baidu.com/s/17GAbZ_YgFJwfZcTpTB_fBA) ； 也可以访问[链接](https://github.com/lonePatient/albert_pytorch) 获取其他albert预训练的中文模型，或者自己训练|[案例](https://github.com/baojunshan/nlp-fluency#bert)|

## 用法
需要安装 `torch`, `transformers`，请自行安装即可。
使用案例可见[example.py](https://github.com/baojunshan/nlp-fluency/blob/master/example.py)

函数：
- score: 负值，越大越好
- perplexity：正值，越小越好

输入：
- 中文
- 可以是一句话，也可以是一段话
- ngram方法需要输入jieba切词的句子，详见下面的案例

测试语料
```python
sentences = [
    "中国人的性情是总喜欢调和折中的，譬如你说，这屋子太暗，须在这里开一个窗，大家一定不允许的。但如果你主张拆掉屋顶他们就来调和，愿意开窗了。",
    "惟将终夜长开眼，报答平生未展眉",
    "我原以为，你身为汉朝老臣，来到阵前，面对两军将士，必有高论。没想到，竟说出如此粗鄙之语！",
    "人生当中成功只是一时的，失败却是主旋律，但是如何面对失败，却把人分成不同的样子，有的人会被失败击垮，有的人能够不断的爬起来继续向前，我想真正的成熟，应该不是追求完美，而是直面自己的缺憾，这才是生活的本质，罗曼罗兰说过，这个世界上只有一种真正的英雄主义，那就是认清生活的真相，并且仍然热爱它。难道向上攀爬的那条路不是比站在顶峰更让人热血澎湃吗？",
    "我在树上游泳。",
    "我在游泳池游泳。",
    "我游泳在游泳池。",
    "尤是为了,更佳大的,念,念,李是彼,更伟大的多,你只会用这种方法解决问题吗!",
]
```

### ngrams
训练模型详见[train_ngramslm.py](https://github.com/baojunshan/nlp-fluency/blob/master/train_ngramslm.py)

由于本模型使用清华摘要数据集训练，缺乏古诗文的语料，导致非白话文的部分ppl偏高，其他都比较准确，在语义的表现也不错，且不受长短句影响。
```python
import jieba
import time
from models import NgramsLanguageModel


start_time = time.time()

model = NgramsLanguageModel.from_pretrained("./thucnews_lm_model")

print(f"Loading ngrams model cost {time.time() - start_time:.3f} seconds.")

for s in sentences:
    ppl = model.perplexity(
        x=jieba.lcut(s),   # 经过切词的句子或段落
        verbose=False,     # 是否显示详细的probability，default=False
    )
    print(f"ppl: {ppl:.5f} # {s}")

print(model.perplexity(jieba.lcut(sentences[-4]), verbose=True))

# Loading ngrams model cost 26.640 seconds.
#
# ppl: 8572.17074 # 中国人的性情是总喜欢调和折中的，譬如你说，这屋子太暗，须在这里开一个窗，大家一定不允许的。但如果你主张拆掉屋顶他们就来调和，愿意开窗了。
# ppl: 660033.44283 # 惟将终夜长开眼，报答平生未展眉
# ppl: 121955.03294 # 我原以为，你身为汉朝老臣，来到阵前，面对两军将士，必有高论。没想到，竟说出如此粗鄙之语！
# ppl: 6831.79220 # 人生当中成功只是一时的，失败却是主旋律，但是如何面对失败，却把人分成不同的样子，有的人会被失败击垮，有的人能够不断的爬起来继续向前，我想真正的成熟，应该不是追求完美，而是直面自己的缺憾，这才是生活的本质，罗曼罗兰说过，这个世界上只有一种真正的英雄主义，那就是认清生活的真相，并且仍然热爱它。难道向上攀爬的那条路不是比站在顶峰更让人热血澎湃吗？
# ppl: 12816.52860 # 我在树上游泳。
# ppl: 7122.96754 # 我在游泳池游泳。
# ppl: 61286.99997 # 我游泳在游泳池。
# ppl: 135742.90546 # 尤是为了,更佳大的,念,念,李是彼,更伟大的多,你只会用这种方法解决问题吗!
#
# ['我', '在'] | 0.00901780
# ['在', '树上'] | 0.00003544
# ['树上', '游泳'] | 0.00000059
# ['游泳', '。'] | 0.00019609
# l score: -13.64571794
# 12816.528602897242
```

### bert
bert总体上比ngrams的方法好，albert虽然速度快，但是效果不理想

```python
from models import MaskedBert, MaskedAlbert

model = MaskedAlbert.from_pretrained("/home/baojunshan/data/pretrained_models/albert_base_zh")

# model = MaskedBert.from_pretrained(
#     path="/home/baojunshan/data/pretrained_models/chinese_bert_wwm_ext_pytorch",
#     device="cpu",  # 使用cpu或者cuda:0，default=cpu
#     sentence_length=50,  # 长句做切句处理，段落会被切成最大不超过该变量的句子集，default=50
# )

for s in sentences:
    ppl = model.perplexity(
        x=" ".join(s),   # 每个字空格隔开或者输入一个list
        verbose=False,     # 是否显示详细的probability，default=False
        temperature=1.0,   # softmax的温度调节，default=1
        batch_size=100,    # 推理时的batch size，可根据cpu或gpu而定，default=100
    )
    print(f"ppl: {ppl:.5f} # {s}")

model.perplexity(sentences[-4], verbose=True)
# model.score(...) # 参数相同

# ppl: 4.20476 # 中国人的性情是总喜欢调和折中的，譬如你说，这屋子太暗，须在这里开一个窗，大家一定不允许的。但如果你主张拆掉屋顶他们就来调和，愿意开窗了。
# ppl: 71.91608 # 惟将终夜长开眼，报答平生未展眉
# ppl: 2.59046 # 我原以为，你身为汉朝老臣，来到阵前，面对两军将士，必有高论。没想到，竟说出如此粗鄙之语！
# ppl: 1.99123 # 人生当中成功只是一时的，失败却是主旋律，但是如何面对失败，却把人分成不同的样子，有的人会被失败击垮，有的人能够不断的爬起来继续向前，我想真正的成熟，应该不是追求完美，而是直面自己的缺憾，这才是生活的本质，罗曼罗兰说过，这个世界上只有一种真正的英雄主义，那就是认清生活的真相，并且仍然热爱它。难道向上攀爬的那条路不是比站在顶峰更让人热血澎湃吗？
# ppl: 10.55426 # 我在树上游泳。
# ppl: 4.38016 # 我在游泳池游泳。
# ppl: 6.56533 # 我游泳在游泳池。
# ppl: 22.52334 # 尤是为了,更佳大的,念,念,李是彼,更伟大的多,你只会用这种方法解决问题吗!
# 我 | 0.00039561
# 在 | 0.96003467
# 树 | 0.00347330
# 上 | 0.42612109
# 游 | 0.95590442
# 泳 | 0.17133135
# 。 | 0.74459237
# l score: -3.39975392
```

### gpt
GPT效果不理想，抛开结果本身，用gpt做通顺度的方法，本身存在一定问题，在预测下一个字的概率时，总是把前面所有的词当做正确的来估计，这会对结果造成偏差。

```python
from models import GPT

model = GPT.from_pretrained(
    path="/home/baojunshan/data/pretrained_models/chinese_gpt2_pytorch",
    device="cpu",
    sentence_length=50
)

for s in sentences:
    ppl = model.perplexity(
        x=" ".join(s),   # 每个字空格隔开或者输入一个list
        verbose=False,     # 是否显示详细的probability，default=False
        temperature=1.0,   # softmax的温度调节，default=1
        batch_size=100,    # 推理时的batch size，可根据cpu或gpu而定，default=100
    )
    print(f"ppl: {ppl:.5f} # {s}")

model.perplexity(sentences[-4], verbose=True)

ppl: 901.41065 # 中国人的性情是总喜欢调和折中的，譬如你说，这屋子太暗，须在这里开一个窗，大家一定不允许的。但如果你主张拆掉屋顶他们就来调和，愿意开窗了。
ppl: 7773.85606 # 惟将终夜长开眼，报答平生未展眉
ppl: 949.33750 # 我原以为，你身为汉朝老臣，来到阵前，面对两军将士，必有高论。没想到，竟说出如此粗鄙之语！
ppl: 906.79251 # 人生当中成功只是一时的，失败却是主旋律，但是如何面对失败，却把人分成不同的样子，有的人会被失败击垮，有的人能够不断的爬起来继续向前，我想真正的成熟，应该不是追求完美，而是直面自己的缺憾，这才是生活的本质，罗曼罗兰说过，这个世界上只有一种真正的英雄主义，那就是认清生活的真相，并且仍然热爱它。难道向上攀 爬的那条路不是比站在顶峰更让人热血澎湃吗？
ppl: 798.38110 # 我在树上游泳。
ppl: 729.68857 # 我在游泳池游泳。
ppl: 469.11313 # 我游泳在游泳池。
ppl: 927.94576 # 尤是为了,更佳大的,念,念,李是彼,更伟大的多,你只会用这种方法解决问题吗!
我 | 0.00924169
在 | 0.00345525
树 | 0.00000974
上 | 0.22259754
游 | 0.00021145
泳 | 0.00004592
。 | 0.00719284
l score: -9.64093376
```

## 计划
- [X] 实现ngrams，gpt，bert mask方法
- [ ] 利用gan的判别器
- [ ] 目前bert和gpt方法实现比较粗糙，速度较慢，之后会加速
- [ ] 虽然bert和gpt的模型可以自己训练完再载入就好，但之后repo也会提供一个train的方法
- [ ] 目前流畅度检测的方法都比较旧了，之后会尝试增加最新的一些方法（又要读论文了

## 引用
```
@misc{GPT2-Chinese,
  author = {Junshan Bao},
  title = {nlp-fluency},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/baojunshan/nlp-fluency}},
}
```