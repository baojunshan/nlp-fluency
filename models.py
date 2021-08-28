from abc import abstractmethod
import math
import os
import json
import pickle
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, BertForPreTraining, BertTokenizer
from transformers import AlbertForPreTraining, AlbertConfig


class ModelMixin:
    def __init__(self, stop_words, sentence_length=50, *args, **kwargs):
        self.stop_words = stop_words or {".", "?", "!", "。", "？", "！"}
        self.stop_words_outer = self.stop_words | {",", "，", ";", "；"}
        self.sentence_length = sentence_length  # 长句切割幅度， 防止bert模型太慢了

    @staticmethod
    @abstractmethod
    def from_pretrained(path, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save(self, path, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, x, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def score(self, x, temperature=1.0, verbose=False, *args, **kwargs):
        raise NotImplementedError

    def perplexity(self, x, temperature=1.0, verbose=False, *args, **kwargs):
        l_score = self.score(x=x, temperature=temperature, verbose=verbose, *args, **kwargs)
        ppl = math.pow(2, -1 * l_score)
        return ppl

    def convert_inputs_to_sentences(self, x):
        if isinstance(x, str):
            x = x.split(" ")
        last_outer_idx = 0
        split_ids = [-1]
        for i, w in enumerate(x):
            if w in self.stop_words_outer:
                last_outer_idx = i
            if i - split_ids[-1] > self.sentence_length:
                if last_outer_idx == split_ids[-1]:
                    raise ValueError(
                        f"Sentence `{''.join(x[last_outer_idx: i + 1])}` is longer than `sentence_length (curr={self.sentence_length})`, please set it larger.")
                split_ids.append(last_outer_idx)
            elif w in self.stop_words:
                split_ids.append(i)
        if split_ids[-1] != len(x) - 1:
            split_ids.append(len(x) - 1)

        sentences = list()
        for start, end in zip(split_ids[:-1], split_ids[1:]):
            sentences.append(x[start + 1: end + 1])
        return sentences


class NgramsLanguageModel(ModelMixin):
    def __init__(self, ngram=2, sentence_length=50, stop_words=None):
        super(NgramsLanguageModel, self).__init__(stop_words=stop_words, sentence_length=sentence_length)
        self.ngram = ngram
        self.model = {self.ngram: dict(), self.ngram - 1: dict()}
        self.corpus_length = 0
        self.token2idx = dict()
        self.idx2token = dict()
        self.token_count = 0

    @staticmethod
    def from_pretrained(path, *args, **kwargs):
        if not os.path.exists(path):
            raise ValueError(f"Did not find the path: {path}, please check.")

        with open(f"{path}/config.json", "r") as f:
            param = json.load(f)
            self = NgramsLanguageModel(
                ngram=int(param["ngram"]),
                sentence_length=int(param["sentence_length"]),
                stop_words=set(param["stop_words"])
            )
            self.corpus_length = int(param["corpus_length"])
            self.token_count = int(param["token_count"])

        with open(f"{path}/vocab.txt", "r") as f:
            for i, t in enumerate(f.read().split("\n")):
                self.token2idx[t] = i
                self.idx2token[i] = t

        with open(f"{path}/model.bin", "r") as f:
            for line in f.readlines():
                line = [int(i) for i in line.strip().split("\t")]
                self.model[len(line) - 1][tuple(line[:-1])] = line[-1]
        return self

    def save(self, path, *args, **kwargs):
        if not os.path.exists(path):
            os.mkdir(path)

        token2idx = sorted(self.token2idx.items(), key=lambda x:[1], reverse=False)
        with open(f"{path}/vocab.txt", "w") as f:
            f.write("\n".join([t for t, i in token2idx]))
        with open(f"{path}/config.json", "w") as f:
            json.dump({
                "corpus_length": self.corpus_length,
                "token_count": self.token_count,
                "sentence_length": self.sentence_length,
                "stop_words": list(self.stop_words),
                "ngram": self.ngram,
            }, f)
        with open(f"{path}/model.bin", "w") as f:
            for n in [self.ngram - 1, self.ngram]:
                for k, v in self.model[n].items():
                    f.write("\t".join([str(i) for i in list(k) + [v]]) + "\n")
        return self

    def train(self, x, *args, **kwargs):
        for line in x:
            if isinstance(line, str):
                line = line.split(" ")

            for w in line:
                if w not in self.token2idx.keys():
                    self.token2idx[w] = self.token_count
                    self.idx2token[self.token_count] = w
                    self.token_count += 1

            self.corpus_length += len(line)

            for i in range(len(line) - self.ngram + 1):
                key = tuple(self.token2idx[t] for t in line[i: i + self.ngram])
                self.model[self.ngram][key] = self.model[self.ngram].get(key, 0) + 1
            if self.ngram > 1:
                for i in range(len(line) - self.ngram + 2):
                    key = tuple(self.token2idx[t] for t in line[i: i + self.ngram - 1])
                    self.model[self.ngram - 1][key] = self.model[self.ngram - 1].get(key, 0) + 1
        return self

    def score(self, x, verbose=False, *args, **kwargs):
        sentences = self.convert_inputs_to_sentences(x)

        word_length = 0
        log_sum_prob = 0
        for sentence in sentences:
            word_length += len(sentence) - self.ngram + 1
            for i in range(len(sentence) - self.ngram + 1):
                words = sentence[i: i + self.ngram]
                prob = self.calc_probability(words)
                if verbose:
                    print(f"{words} | {prob:.8f}")
                log_sum_prob += math.log(prob, 2)

        if word_length == 0:
            l_score = 0
        else:
            l_score = log_sum_prob / word_length
        if verbose:
            print(f"l score: {l_score:.8f}")
        return l_score

    def calc_probability(self, words):
        key = tuple(self.token2idx.get(t, -1) for t in words)
        words_freq = self.model[self.ngram].get(key, 0)
        if self.ngram == 1:
            return (words_freq + 1) / (self.corpus_length + len(self.model[self.ngram].keys()))
        return (words_freq + 1) / (self.model[self.ngram - 1].get(key[:-1], 0) + len(self.model[self.ngram - 1].keys()))


class MaskedBert(ModelMixin):
    def __init__(self, stop_words=None, sentence_length=50, device="cpu"):
        super(MaskedBert, self).__init__(stop_words=stop_words, sentence_length=sentence_length)
        self.model = None
        self.tokenizer = None
        self.mask_id = -1
        self.device = device

    @staticmethod
    def from_pretrained(path, sentence_length=50, device="cpu", stop_words=None, *args, **kwargs):
        model = BertForPreTraining.from_pretrained(path)
        tokenizer = BertTokenizer.from_pretrained(path)
        self = MaskedBert(device=device, stop_words=stop_words, sentence_length=sentence_length)
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.mask_id = int(tokenizer.convert_tokens_to_ids("[MASK]"))

        return self

    def save(self, path, *args, **kwargs):
        pass

    def train(self, x, *args, **kwargs):
        pass

    def score(self, x, temperature=1.0, batch_size=100, verbose=False, *args, **kwargs):
        self.model.eval()

        sentences = self.convert_inputs_to_sentences(x)
        all_probability = list()
        all_words = list()
        for sentence in sentences:
            inputs = self.tokenizer("".join(sentence), return_tensors="pt")
            input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs[
                "attention_mask"]
            origin_ids = input_ids[0][1: -1]
            length = input_ids.shape[-1] - 2

            batch_indice = list()
            for i in range(length // batch_size):
                batch_indice.append([i * batch_size, (i + 1) * batch_size])
            if length % batch_size != 0:
                batch_indice.append([batch_size * (length // batch_size), length])

            for start, end in batch_indice:
                ids_list = list()
                for i in range(start, end):
                    tmp = input_ids.clone()
                    tmp[0][i + 1] = self.mask_id
                    ids_list.append(tmp)
                new_input_ids = torch.cat(ids_list, dim=0)
                new_attention_mask = attention_mask.expand(end - start, length + 2)
                new_token_type_ids = token_type_ids.expand(end - start, length + 2)
                inputs = {
                    'input_ids': new_input_ids.to(self.device),
                    'token_type_ids': new_token_type_ids.to(self.device),
                    'attention_mask': new_attention_mask.to(self.device)
                }
                outputs = self.model(**inputs).prediction_logits
                outputs = F.softmax(outputs / temperature, dim=-1).detach().cpu().numpy()
                probability = [outputs[i][start + i + 1][ids] for i, ids in enumerate(origin_ids[start: end])]
                all_probability += probability
                all_words += self.tokenizer.convert_ids_to_tokens(origin_ids[start: end])

        if len(all_probability) == 0:
            l_score = 0
        else:
            l_score = sum([math.log(p, 2) for p in all_probability]) / len(all_probability)

        if verbose:
            words = list()
            for s in sentences:
                words += s
            for word, prob in zip(all_words, all_probability):
                print(f"{word} | {prob:.8f}")
            print(f"l score: {l_score:.8f}")

        return l_score


class MaskedAlbert(MaskedBert):
    def __init__(self, stop_words=None, sentence_length=50, device="cpu"):
        super(MaskedAlbert, self).__init__(stop_words=stop_words, sentence_length=sentence_length, device=device)

    @staticmethod
    def from_pretrained(path, stop_words=None, sentence_length=50, device="cpu", *args, **kwargs):
        bert_config = AlbertConfig.from_pretrained(path)
        model = AlbertForPreTraining(config=bert_config)
        tokenizer = BertTokenizer.from_pretrained(path)
        self = MaskedAlbert(device=device, sentence_length=sentence_length, stop_words=stop_words)
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.mask_id = int(tokenizer.convert_tokens_to_ids("[MASK]"))
        return self


class GPT(ModelMixin):
    def __init__(self, device="cpu", stop_words=None, sentence_length=50):
        super(GPT, self).__init__(stop_words=stop_words, sentence_length=sentence_length)
        self.model = None
        self.tokenizer = None
        self.device = device

    @staticmethod
    def from_pretrained(path, device="cpu", stop_words=None, sentence_length=50, *args, **kwargs):
        model = GPT2LMHeadModel.from_pretrained(path)
        tokenizer = BertTokenizer.from_pretrained(path)
        self = GPT(device=device, stop_words=stop_words, sentence_length=sentence_length)
        self.model = model.to(device)
        self.tokenizer = tokenizer
        return self

    def save(self, path, *args, **kwargs):
        pass

    def train(self, x, *args, **kwargs):
        pass

    def score(self, x, temperature=1.0, window=100, verbose=False, *args, **kwargs):
        self.model.eval()

        sentences = self.convert_inputs_to_sentences(x)

        all_probability = list()
        all_words = list()
        for sentence in sentences:
            input_ids = self.tokenizer("".join(sentence), return_tensors="pt")["input_ids"]
            origin_ids = input_ids[0][1: -1]
            for i in range(len(origin_ids)):
                text = self.tokenizer.convert_ids_to_tokens(origin_ids[max(0, i - window):i])
                inputs = self.tokenizer("".join(text), return_tensors="pt").to(self.device)
                outputs = self.model(**inputs).logits[0, -1, :]
                outputs = F.softmax(outputs / temperature, dim=-1).detach().cpu().numpy()
                probability = outputs[origin_ids[i]]
                all_probability.append(probability)
                all_words += self.tokenizer.convert_ids_to_tokens([origin_ids[i]])

        if len(all_probability) == 0:
            l_score = 0
        else:
            l_score = sum([math.log(p, 2) for p in all_probability]) / len(all_probability)
        if verbose:
            words = list()
            for s in sentences:
                words += s
            for word, prob in zip(all_words, all_probability):
                print(f"{word} | {prob:.8f}")
            print(f"l score: {l_score:.8f}")

        return l_score

