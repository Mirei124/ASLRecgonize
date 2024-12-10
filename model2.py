import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 384
MAX_SEQ_LEN = 96
INPUT_FEATURES = 122


class CNNEmbedding(nn.Module):
    def __init__(self, input_channels=INPUT_FEATURES, output_channels=D_MODEL, max_seq_len=MAX_SEQ_LEN) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv1 = nn.Conv1d(input_channels, output_channels, 3, 1, 1)
        self.ln1 = nn.LayerNorm((output_channels, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """利用1D-CNN提取特征.

        Args:
            x: Tensor (batch, feature, seq)

        Returns:
            返回一个Tensor (batch, max_seq, feature)
        """
        # if x.ndim != 3:
        #     x = x.unsqueeze(0)
        x = F.relu(self.ln1(self.conv1(x)))
        # print("after conv")
        # print(x[0])
        return x.permute((0, 2, 1))

    def __str__(self) -> str:
        return f"CNNEmbedding: Convert (batch, {self.input_channels}, seq_len) to (batch, seq_len, {self.output_channels}), then pad by 0 to (batch, {self.max_seq_len}, {self.output_channels})"


class MLP(nn.Module):
    def __init__(self, input_channels=D_MODEL, num_classes=250, hidden_dim=1024) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.a1 = nn.ReLU()
        self.ln1 = nn.LayerNorm((hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """根据TransformerEncoder的输出预测最终类别.

        Args:
            x: Tensor (batch, feature)

        Returns:
            返回一个Tensor, 形状为 (batch, num_classes)
        """
        x = self.a1(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        return x

    def __str__(self) -> str:
        return f"MLP: (batch, {self.input_channels}) -> (batch, {self.num_classes})"


class TFEncoder(nn.Module):
    def __init__(self, max_seq_len=MAX_SEQ_LEN, model_dim=D_MODEL) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.model_dim = model_dim
        # self.pos_param = nn.Parameter(torch.zeros(1, max_seq_len + 1, model_dim))
        self.pos_param = get_cosine_positional_encoding(max_seq_len + 1, model_dim).to("cuda")
        self.cls_param = nn.Parameter(torch.zeros(1, model_dim))
        # input (batch, seq, feature)
        encoder_layer = nn.TransformerEncoderLayer(model_dim, 8, batch_first=True, norm_first=True)
        self.tf_encoder = nn.TransformerEncoder(encoder_layer, 6, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入添加位置编码和cls token, 使用TransformerEncoder对输入进行处理, 输出cls token对应的输出.

        Args:
            x (torch.Tensor): 形状为(batch, seq, feature)

        Returns:
            返回一个Tensor, 形状为(batch, feature)
        """
        x_cls = torch.cat([self.cls_param.repeat((x.shape[0], 1, 1)), x], dim=1)
        x_pos_enc = x_cls + self.pos_param

        output = self.tf_encoder(x_pos_enc)
        return output[:, 0, :]

    def __str__(self) -> str:
        return f"TransformerEncoder: (batch, {self.max_seq_len}, {self.model_dim}) -> (batch, {self.model_dim})"


class MyModel(nn.Module):
    def __init__(self, input_channels=122, model_dim=384, max_seq_len=96, num_classes=250, hidden_dim=1024):
        super().__init__()
        self.input_channels = input_channels
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.cnn = CNNEmbedding(input_channels, model_dim, max_seq_len)
        self.tfe = TFEncoder(max_seq_len, model_dim)
        self.mlp = MLP(model_dim, num_classes, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x -> CNNEmbedding -> TransformerEncoder -> MLP -> class.

        Args:
            x: Tensor (batch, feature, seq)

        Returns:
            x TransformerEncoder 提取的特征 (batch, model_dim)
            y 分类结果 (batch, num_classes)
        """
        x = self.cnn(x)
        # print("after cnn")
        # print(x[0])
        x = self.tfe(x)
        # B, model_dim 384
        # print("after tfe")
        # print(x[0])
        y = self.mlp(x)
        # print("after mlp")
        # print(x[0])
        return x, y


class MixUpModel(nn.Module):
    def __init__(self, gloss_feature=None, num_classes=250, text_token_len=300, model_dim=384, hidden_dim=1024) -> None:
        super().__init__()
        self.gloss_feature = gloss_feature
        self.num_classes = num_classes
        self.text_token_len = text_token_len
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        # (num_classes, text_token_len) -> (num_classes, model_dim)
        self.fc1 = nn.Linear(text_token_len, model_dim)
        # (num_classes, model_dim) -> (num_classes, num_classes)
        self.mlp = MLP(model_dim, num_classes, hidden_dim)

    def forward(self, tf_feature: torch.Tensor) -> torch.Tensor:
        """融合视频和文字标签特征，输出计算$L_{IMM}$.

        Args:
            tf_feature: TransformerEncoder输出的特征 (batch, model_dim)

        Returns:
            结果 (num_classes, num_classes)
        """
        tf_feature = tf_feature.unsqueeze(dim=1)
        # tf_feature (batch, 1, model_dim)
        gloss_proj = F.relu(self.fc1(self.gloss_feature))
        x = gloss_proj + tf_feature.repeat(1, self.num_classes, 1)
        x = self.mlp(x)
        return x

    def __str__(self) -> str:
        return f"MixUpModel: (batch, model_dim) -> (batch, num_classes {self.num_classes}, text_token_len {self.text_token_len})"


def get_real_gloss_feature() -> torch.Tensor:
    # (num_classes, text_token_len)
    with open("./word_to_vector.json", "r") as fp:
        w2v = json.loads(fp.read())
    with open("./data/sign_to_prediction_index_map.json", "r") as fp:
        sign2idx = json.loads(fp.read())
    idx2sign = {v: k for k, v in sign2idx.items()}
    word_list = []
    for i in range(250):
        word_list.append(torch.tensor(w2v[idx2sign[i]]))
    return torch.stack(word_list)


def get_real_soft_label(gloss_feature: torch.Tensor, epsilon=0.2, tau=0.5) -> torch.Tensor:
    a = gloss_feature.norm(p=2, dim=1).view(-1, 1)
    b = a @ a.T
    soft_label_list = []
    for i in range(250):
        row = b[i]
        row[i] = 0.0
        row = torch.exp(row / tau)
        row = row / row.sum()
        weight = epsilon * row
        weight[i] = 1 - epsilon
        soft_label_list.append(weight)
    # (num_classes, num_classes)
    return torch.stack(soft_label_list)


def get_dummy_gloss_feature() -> torch.Tensor:
    return torch.randn(250, 300)


def get_dummy_soft_label() -> torch.Tensor:
    return torch.randn(250)


def get_cosine_positional_encoding(max_seq_len: int, model_dim: int) -> torch.Tensor:
    position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-np.log(10000.0) / model_dim))

    pe = torch.zeros(max_seq_len, model_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


class SoftLabel:
    def __init__(self, all_soft_label: torch.Tensor):
        self.all_soft_label = all_soft_label

    def generate_soft_label(self, label: torch.Tensor) -> torch.Tensor:
        # label: (batch)
        result = []
        for i in range(label.shape[0]):
            result.append(self.all_soft_label[label[i].item()])
        return torch.stack(result)


class IMMLabel:
    def __init__(self, num_classes=250):
        self.num_classes = num_classes
        self.cache = {}

    def _generate_one_label(self, idx: int) -> torch.Tensor:
        if idx in self.cache.keys():
            return self.cache[idx]
        a = torch.eye(self.num_classes) * 0.5
        for i in range(self.num_classes):
            a[i, idx] = 0.5
        a[idx, idx] = 1.0
        self.cache[idx] = a
        return a

    def generate_imm_label(self, idx: torch.Tensor, num_classes=250) -> torch.Tensor:
        # idx: (batch)
        result = []
        for i in range(idx.shape[0]):
            result.append(self._generate_one_label(idx[i].item()))
        return torch.stack(result)


if __name__ == "__main__":
    h = get_real_soft_label(get_real_gloss_feature())
    print(h.shape)
    print(h[11])
    print(h[11].sum())
    print(h[11].argmax())
    exit()
    # (batch, feature, seq)
    a = torch.randn((2, 122, 96))
    b = CNNEmbedding()(a)
    print(f"after CNNEmbedding {b.shape}")
    # (batch, max_seq, model_dim)
    c = TFEncoder()(b)
    print(f"after TF {c.shape}")
    d = MLP()(c)
    print(f"after MLP {d.shape}")

    model1 = MyModel()
    model2 = MixUpModel()
    e, f = model1(a)
    print(f"MyModel output {f.shape}")
    g = model2(e)
    print(f"MixUpModel output {g.shape}")

    # print(get_cosine_positional_encoding(97, 284).shape)
