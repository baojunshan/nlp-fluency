import glob
import jieba
import time
from tqdm import tqdm
from models import NgramsLanguageModel


model = NgramsLanguageModel(ngram=2)

path = glob.glob("/home/baojunshan/data/titles/THUNews/*/*.txt")

data = list()
for i, p in enumerate(tqdm(path)):
    if i % 100000 == 0 and i != 0:
        print(f"{i}: save model")
        model.save("thucnews.lm")
    if i % 100 == 0 and len(data) > 0:
        model.train(data)
        data = list()
    with open(p, "r") as f:
        for j, line in enumerate(f.readlines()):
            if j > 0:
                data.append(jieba.lcut(line.strip()))

print("final save model")
model.save("thucnews.lm")
print("done!")

start_time = time.time()
model2 = NgramsLanguageModel.from_pretrained("thucnews.lm")
print("load model", time.time() - start_time)


print(len(model.token2idx.keys()))

print(model.perplexity(jieba.lcut("这是哪个这个啊这个"), verbose=True))
print(model.perplexity(jieba.lcut("问渠哪得清如许，为有源头活水来"), verbose=True))
print(model.perplexity(jieba.lcut("满脑子新花招的宝瓶男子，事事充满着科学精神"), verbose=True))

