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

# ------------------------------
# NgramsLanguageModel
# ------------------------------
import time
import jieba
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
# model.score(...) # 参数相同
exit()


# -----------------------------
# bert or albert
# -----------------------------
from models import MaskedBert, MaskedAlbert

# model = MaskedAlbert.from_pretrained("/home/baojunshan/data/pretrained_models/albert_base_zh")

# model = MaskedBert.from_pretrained(
#     path="/home/baojunshan/data/pretrained_models/chinese_bert_wwm_ext_pytorch",
#     device="cpu",  # 使用cpu或者cuda:0，default=cpu
#     sentence_length=50,  # 长句做切句处理，段落会被切成最大不超过该变量的句子集，default=50
# )
#
# for s in sentences:
#     ppl = model.perplexity(
#         x=" ".join(s),   # 每个字空格隔开或者输入一个list
#         verbose=False,     # 是否显示详细的probability，default=False
#         temperature=1.0,   # softmax的温度调节，default=1
#         batch_size=100,    # 推理时的batch size，可根据cpu或gpu而定，default=100
#     )
#     print(f"ppl: {ppl:.5f} # {s}")
#
# model.perplexity(sentences[-4], verbose=True)
# model.score(...) # 参数相同

# --------------------------------
# GPT
# --------------------------------
# from models import GPT
#
# model = GPT.from_pretrained(
#     path="/home/baojunshan/data/pretrained_models/chinese_gpt2_pytorch",
#     device="cpu",
#     sentence_length=50
# )
#
# for s in sentences:
#     ppl = model.perplexity(
#         x=" ".join(s),   # 每个字空格隔开或者输入一个list
#         verbose=False,     # 是否显示详细的probability，default=False
#         temperature=1.0,   # softmax的温度调节，default=1
#         batch_size=100,    # 推理时的batch size，可根据cpu或gpu而定，default=100
#     )
#     print(f"ppl: {ppl:.5f} # {s}")
#
# model.perplexity(sentences[-4], verbose=True)



