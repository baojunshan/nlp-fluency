from models import Gan
from torch.utils.data import Dataset, DataLoader
import glob
import random
from tqdm import tqdm


class THUDataset(Dataset):
    def __init__(self, path, max_len, skip_error=False):
        super(THUDataset, self).__init__()
        self.stop_words = {".", "?", "!", "。", "？", "！"}
        self.stop_words_outer = self.stop_words | {",", "，", ";", "；", "~", "……"}
        self.max_len = max_len
        self.skip_error = skip_error

        path = glob.glob(f"{path}/*/*.txt")
        random.shuffle(path)
        self.id2path = {i: p for i, p in enumerate(path)}
        self.meta = self.init_meta()

    @staticmethod
    def get_article(path):
        with open(path, "r", encoding="utf-8") as f:
            s = "".join([line.strip() for j, line in enumerate(f.readlines()) if j > 0])
        return s

    def init_meta(self):
        meta = list()
        for i, p in tqdm(self.id2path.items(), desc="Init meta"):
            s = self.get_article(p)
            ids = self.convert_inputs_to_split_ids(s)
            for start, end in zip(ids[:-1], ids[1:]):
                meta.append([i, start, end])
        random.shuffle(meta)
        print(f"Totle data size is: {len(meta)}")
        return meta

    def convert_inputs_to_split_ids(self, x):
        last_outer_idx = 0
        split_ids = [-1]
        for i, w in enumerate(x):
            if w in self.stop_words_outer:
                last_outer_idx = i
            if i - split_ids[-1] > self.max_len:
                if last_outer_idx == split_ids[-1] and not self.skip_error:
                    raise ValueError(
                        f"Sentence `{''.join(x[last_outer_idx: i + 1])}` is longer than `max_len (curr={self.max_len})`, please set it larger.")
                else:
                    split_ids.append(last_outer_idx)
            elif w in self.stop_words:
                split_ids.append(i)
        if split_ids[-1] != len(x) - 1:
            split_ids.append(len(x) - 1)

        return [i + 1 for i in split_ids]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, item):
        path_idx, split_start, split_end = self.meta[item]
        content = self.get_article(self.id2path[path_idx])
        return content[split_start: split_end]


batch_size = 512
max_len = 50

dataset = THUDataset(
    path="/home/baojunshan/data/THUNews",
    max_len=max_len,
    skip_error=True,
)
train_data = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=16,
    drop_last=True
)

model = Gan(
    vocab_path="/home/baojunshan/data/pretrained_models/chinese_bert_wwm_ext_pytorch/vocab.txt",
    in_dim=100,
    emb_dim=300,
    kernel_with=5,
    gcnn_layers=6,
    gcnn_channels=32,
    gcnn_blocks=2,
    gcnn_out_dim=10,
    device="cuda:0",
    stop_words=None,
    max_len=max_len
)

param_num = model.calc_params()
print(f"Model Params: [Generator] {param_num['Generator'] / 1e6}M [Discriminator] {param_num['Discriminator'] / 1e6}M.")

model.train(
    x=train_data,
    epoch=30,
    batch_size=batch_size,
    lr_g=1e-3,
    lr_d=1e-3,
    n_step_per_discriminator=10,
    n_step_per_generator=1,
    n_epoch_per_evaluate=1,
    save_path="./gan_models",
    n_epoch_to_save=1,
)
