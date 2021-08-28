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
        model.save("thucnews_lm_model")
    if i % 100 == 0 and len(data) > 0:
        model.train(data)
        data = list()
    with open(p, "r") as f:
        for j, line in enumerate(f.readlines()):
            if j > 0:
                data.append(jieba.lcut(line.strip()))

print("final save model")
model.save("thucnews_lm_model")
print("done!")

start_time = time.time()
model2 = NgramsLanguageModel.from_pretrained("thucnews_lm_model")
print("load model", time.time() - start_time)

print(len(model.token2idx.keys()), len(model2.token2idx.keys()))
print(model.token_count, model2.token_count)
print(model.corpus_length, model2.corpus_length)

print(model2.perplexity(jieba.lcut("问渠哪得清如许，为有源头活水来"), verbose=True))

