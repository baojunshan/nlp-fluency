from abc import abstractmethod
import math
import time
import os
import json
from shutil import copyfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, BertForPreTraining, BertTokenizer
from transformers import AlbertForPreTraining, AlbertConfig


class ModelMixin:
    def __init__(self, stop_words, max_len=50, *args, **kwargs):
        self.stop_words = stop_words or {".", "?", "!", "。", "？", "！"}
        self.stop_words_outer = self.stop_words | {",", "，", ";", "；"}
        self.max_len = max_len  # 长句切割幅度， 防止bert模型太慢了

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
            if i - split_ids[-1] > self.max_len:
                if last_outer_idx == split_ids[-1]:
                    raise ValueError(
                        f"Sentence `{''.join(x[last_outer_idx: i + 1])}` is longer than `sentence_length (curr={self.max_len})`, please set it larger.")
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
    def __init__(self, ngram=2, max_len=50, stop_words=None):
        super(NgramsLanguageModel, self).__init__(stop_words=stop_words, max_len=max_len)
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
                max_len=int(param["max_len"]),
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

        token2idx = sorted(self.token2idx.items(), key=lambda x: [1], reverse=False)
        with open(f"{path}/vocab.txt", "w") as f:
            f.write("\n".join([t for t, i in token2idx]))
        with open(f"{path}/config.json", "w") as f:
            json.dump({
                "corpus_length": self.corpus_length,
                "token_count": self.token_count,
                "max_len": self.max_len,
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
    def __init__(self, stop_words=None, max_len=50, device="cpu"):
        super(MaskedBert, self).__init__(stop_words=stop_words, max_len=max_len)
        self.model = None
        self.tokenizer = None
        self.mask_id = -1
        self.device = device

    @staticmethod
    def from_pretrained(path, max_len=50, device="cpu", stop_words=None, *args, **kwargs):
        model = BertForPreTraining.from_pretrained(path)
        tokenizer = BertTokenizer.from_pretrained(path)
        self = MaskedBert(device=device, stop_words=stop_words, max_len=max_len)
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
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

        l_score = 0
        if len(all_probability) > 0:
            l_score = sum([math.log(p, 2) for p in all_probability]) / len(all_probability)

        if verbose:
            for word, prob in zip(all_words, all_probability):
                print(f"{word} | {prob:.8f}")
            print(f"l score: {l_score:.8f}")

        return l_score


class MaskedAlbert(MaskedBert):
    def __init__(self, stop_words=None, max_len=50, device="cpu"):
        super(MaskedAlbert, self).__init__(stop_words=stop_words, max_len=max_len, device=device)

    @staticmethod
    def from_pretrained(path, stop_words=None, max_len=50, device="cpu", *args, **kwargs):
        bert_config = AlbertConfig.from_pretrained(path)
        model = AlbertForPreTraining(config=bert_config)
        tokenizer = BertTokenizer.from_pretrained(path)
        self = MaskedAlbert(device=device, max_len=max_len, stop_words=stop_words)
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.mask_id = int(tokenizer.convert_tokens_to_ids("[MASK]"))
        return self


class GPT(ModelMixin):
    def __init__(self, device="cpu", stop_words=None, max_len=50):
        super(GPT, self).__init__(stop_words=stop_words, max_len=max_len)
        self.model = None
        self.tokenizer = None
        self.device = device

    @staticmethod
    def from_pretrained(path, device="cpu", stop_words=None, max_len=50, *args, **kwargs):
        model = GPT2LMHeadModel.from_pretrained(path)
        tokenizer = BertTokenizer.from_pretrained(path)
        self = GPT(device=device, stop_words=stop_words, max_len=max_len)
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
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

        l_score = 0
        if len(all_probability) > 0:
            l_score = sum([math.log(p, 2) for p in all_probability]) / len(all_probability)
        if verbose:
            for word, prob in zip(all_words, all_probability):
                print(f"{word} | {prob:.8f}")
            print(f"l score: {l_score:.8f}")

        return l_score


class GatedCNN(nn.Module):
    def __init__(self, out_dim, seq_len, kernel_shape, out_channels=32, n_layers=10, res_block_count=5):
        super(GatedCNN, self).__init__()
        self.res_block_count = res_block_count
        self.seq_len = seq_len

        padding_size = kernel_shape[0] // 2

        self.conv_0 = nn.Conv2d(1, out_channels, kernel_shape, padding=(padding_size, 0))
        self.bias_0 = nn.Parameter(torch.randn(1, out_channels, 1, 1))
        self.conv_gate_0 = nn.Conv2d(1, out_channels, kernel_shape, padding=(padding_size, 0))
        self.bias_gate_0 = nn.Parameter(torch.randn(1, out_channels, 1, 1))

        self.conv = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, (kernel_shape[0], 1), padding=(padding_size, 0)) \
            for _ in range(n_layers)
        ])
        self.conv_gate = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, (kernel_shape[0], 1), padding=(padding_size, 0)) \
            for _ in range(n_layers)
        ])
        self.bias = nn.ParameterList([
            nn.Parameter(torch.randn(1, out_channels, 1, 1)) \
            for _ in range(n_layers)
        ])
        self.bias_gate = nn.ParameterList([
            nn.Parameter(torch.randn(1, out_channels, 1, 1)) \
            for _ in range(n_layers)
        ])

        self.linear = nn.Linear(out_channels * seq_len, out_dim)

    def forward(self, x):
        # input x: b, seq_len, embed_dim
        x = x.unsqueeze(1)  # b, c, seq_len, embed_dim
        a = self.conv_0(x)
        a = a + self.bias_0
        b = self.conv_gate_0(x)
        b = b + self.bias_gate_0
        h = a * F.sigmoid(b)
        res_input = h

        for i, (conv, conv_gate) in enumerate(zip(self.conv, self.conv_gate)):
            a = conv(h) + self.bias[i]
            b = conv_gate(h) + self.bias_gate[i]
            h = a * F.sigmoid(b)  # b, c, seq_len, 1
            if (i + 1) % self.res_block_count == 0:  # size of each residual block
                h += res_input
                res_input = h

        h = h.view(h.shape[0], -1)
        out = self.linear(h)
        out = F.relu(out, inplace=True)
        return out


class Generator(nn.Module):
    def __init__(self, in_dim, seq_len, vocab_size):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.linear0 = nn.Linear(in_dim, in_dim * 2)
        self.bn = nn.BatchNorm1d(in_dim * 2)
        self.linear1 = nn.Linear(in_dim * 2, seq_len * vocab_size)

    def forward(self, x):
        x = self.linear0(x)
        x = F.relu(x, inplace=True)
        x = self.bn(x)
        x = self.linear1(x).view(-1, self.seq_len, self.vocab_size)  # b, seq_len, emb_dim
        x = F.gumbel_softmax(x, hard=True)
        return x


class Discriminator(nn.Module):
    def __init__(self, seq_len, vocab_size, emb_dim, kernel_width=5,
                 gcnn_layers=10, gcnn_channels=32, gcnn_blocks=5, gcnn_out_dim=10):
        super(Discriminator, self).__init__()
        self.embedding = nn.Linear(vocab_size, emb_dim)
        self.gcnn = GatedCNN(
            out_dim=gcnn_out_dim,
            seq_len=seq_len,
            kernel_shape=(kernel_width, emb_dim),
            out_channels=gcnn_channels,
            n_layers=gcnn_layers,
            res_block_count=gcnn_blocks,
        )
        self.classification = nn.Linear(gcnn_out_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.gcnn(x)
        x = self.classification(x)
        x = self.activation(x)
        return x


class Gan(ModelMixin):
    def __init__(self, vocab_path, in_dim=100, emb_dim=300, kernel_with=5,
                 gcnn_layers=10, gcnn_channels=32, gcnn_blocks=5, gcnn_out_dim=10,
                 device="cpu", stop_words=None, max_len=50):
        super(Gan, self).__init__(stop_words=stop_words, max_len=max_len)
        self.device = device
        self.vocab_path = vocab_path
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        self.generator = Generator(in_dim=in_dim, seq_len=max_len, vocab_size=self.tokenizer.vocab_size)
        self.discriminator = Discriminator(
            seq_len=max_len, vocab_size=self.tokenizer.vocab_size, emb_dim=emb_dim, kernel_width=kernel_with,
            gcnn_layers=gcnn_layers, gcnn_channels=gcnn_channels, gcnn_blocks=gcnn_blocks, gcnn_out_dim=gcnn_out_dim
        )
        self.config = {
            "in_dim": in_dim, "emb_dim": emb_dim, "kernel_with": kernel_with,
            "gcnn_layers": gcnn_layers, "gcnn_channels": gcnn_channels,
            "gcnn_blocks": gcnn_blocks, "gcnn_out_dim": gcnn_out_dim,
            "max_len": max_len
        }

    def calc_params(self):
        return {
            "Generator": sum(p.numel() for p in self.generator.parameters()),
            "Discriminator": sum(p.numel() for p in self.discriminator.parameters())
        }

    @staticmethod
    def from_pretrained(path, device="cpu", stop_words=None, sentence_length=50, *args, **kwargs):
        config = json.load(open(f"{path}/config.json", "r"))
        self = Gan(**config, vocab_path=f"{path}/vocab.txt")
        self.device = device
        ckpt = torch.load(f"{path}/model.bin")
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        return self

    def save(self, path, *args, **kwargs):
        if not os.path.exists(path):
            os.mkdir(path)
        copyfile(self.vocab_path, os.path.join(path, "vocab.txt"))
        json.dump(self.config, open(f"{path}/config.json", "w"))
        torch.save({
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }, f"{path}/model.bin")

    def preprocessing(self, sentences):
        ids = list()
        for text in sentences:
            text_ids = self.tokenizer.encode(text, max_length=self.max_len, truncation='only_first')
            text_ids = text_ids[1:-1]
            text_ids = text_ids[:self.max_len] + [0] * (self.max_len - len(text_ids))
            ids.append(text_ids)
        ids = F.one_hot(torch.tensor(ids), num_classes=self.tokenizer.vocab_size).type(torch.FloatTensor)
        return ids

    def train(self, x, epoch=5, batch_size=8, lr_g=1e-3, lr_d=1e-3,
              n_step_per_discriminator=1, n_step_per_generator=1, n_epoch_per_evaluate=1,
              save_path=None, n_epoch_to_save=1, *args, **kwargs):

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.loss = nn.BCELoss().to(self.device)
        self.optimizer_g = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=lr_g,
            betas=(0.5, 0.999)
        )
        self.optimizer_d = torch.optim.Adam(
            params=self.discriminator.parameters(),
            lr=lr_d,
            betas=(0.5, 0.999)
        )

        for ep in range(epoch):
            sum_time = 0
            cnt_time = 0
            d_loss, g_loss = None, None

            self.generator.train()
            self.discriminator.train()
            for i, data in enumerate(x):
                start_time = time.time()
                # tokenize
                data = self.preprocessing(data)

                valid = torch.autograd.Variable(torch.ones(data.shape[0], 1), requires_grad=False).to(self.device)
                fake = torch.autograd.Variable(torch.zeros(data.shape[0], 1), requires_grad=False).to(self.device)

                real_data = data.to(self.device)

                z = torch.normal(0, 1, (data.shape[0], self.config["in_dim"])).to(self.device)
                gen_data = self.generator(z)

                # -----------------
                #  Train discriminator
                # -----------------
                if i % n_step_per_discriminator == 0:
                    self.optimizer_d.zero_grad()  # 以前的梯度清空
                    real_loss = self.loss(self.discriminator(real_data), valid)
                    fake_loss = self.loss(self.discriminator(gen_data.detach()), fake)  # 不更新生成器
                    d_loss = (real_loss + fake_loss) / 2

                    d_loss.backward()  # 梯度下降
                    self.optimizer_d.step()  # 更新优化器

                # -----------------
                #  Train Generator
                # -----------------
                if i % n_step_per_generator == 0:
                    self.optimizer_g.zero_grad()
                    g_loss = self.loss(self.discriminator(gen_data), valid)
                    g_loss.backward()
                    self.optimizer_g.step()

                sum_time += time.time() - start_time
                cnt_time += 1
                print(
                    f"\r[Epoch {ep + 1:03}/{epoch:03}]",
                    f"Batch {i + 1:05}/{len(x):05} [{sum_time:.2f}s/{(sum_time / cnt_time) * len(x) - sum_time:.2f}s - {sum_time / cnt_time:.3f} s/it] ",
                    f"D loss: {d_loss.item():.5f} G loss: {g_loss.item():.5f}",
                    end=""
                )
            if d_loss is not None and g_loss is not None:
                print(
                    f"\r[Epoch {ep + 1}/{epoch}]",
                    f"D loss {d_loss.item():5f} G loss {g_loss.item():5f}",
                    f"Time {sum_time:.2f}s"
                )

            if (ep + 1) % n_epoch_per_evaluate == 0:
                self.generator.eval()
                self.discriminator.eval()
                z = torch.normal(0, 1, (5, self.config["in_dim"])).to(self.device)
                gen = self.generator(z)
                score = self.discriminator(gen).detach().cpu().numpy()
                ids = torch.argmax(gen, dim=-1).detach().cpu().numpy()
                for i in range(5):
                    print(score[i, 0], "".join(self.tokenizer.convert_ids_to_tokens(ids[i, :])))

            if (ep + 1) % n_epoch_to_save == 0 and save_path is not None:
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                self.save(os.path.join(save_path, f"epoch_{ep}"))

    def score(self, x, temperature=1.0, verbose=False, *args, **kwargs):
        self.discriminator.eval()
        sentences = self.convert_inputs_to_sentences(x)
        data = self.preprocessing(sentences).to(self.device)
        scores = self.discriminator(data).detach().cpu().numpy()

        probabilities = list()
        for i in range(len(sentences)):
            score = float(scores[i, 0])
            probabilities += [score] * len(sentences[i])

        l_score = 0
        if len(probabilities) > 0:
            l_score = sum([math.log(p, 2) for p in probabilities]) / len(probabilities)
        if verbose:
            for word, prob in zip(x, all_probabilities):
                print(f"{word} | {prob:.8f}")
            print(f"l score: {l_score:.8f}")
        return l_score
