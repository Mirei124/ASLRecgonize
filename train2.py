import os
import time
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from dataset2 import ASLDataset
from model2 import IMMLabel, MixUpModel, MyModel, SoftLabel, get_real_gloss_feature, get_real_soft_label

batch_size = 64
initial_lr = 0.001
warmup_epoch = 5
total_epoch = 150

model_name = "test1"
result_path = "results2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lr_scheduler(epoch) -> float:
    if epoch < warmup_epoch:
        return epoch * initial_lr / warmup_epoch
    else:
        return initial_lr * ((1 - (epoch - warmup_epoch) / (total_epoch - warmup_epoch)) ** 0.9)


writer = SummaryWriter(comment=model_name)

asl_dataset = ASLDataset(is_train=True)
asl_dataloader = DataLoader(asl_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

test_dataset = ASLDataset(is_train=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=1)

imm_label = IMMLabel()
gloss_feature = get_real_gloss_feature().to(device)
all_soft_label = get_real_soft_label(gloss_feature).to(device)
soft_label_converter = SoftLabel(all_soft_label)
model1 = MyModel(input_channels=122, model_dim=128, max_seq_len=96, num_classes=250, hidden_dim=512).to(device)
model2 = MixUpModel(gloss_feature=gloss_feature, num_classes=250, text_token_len=300, model_dim=128, hidden_dim=512).to(
    device
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(chain(model1.parameters(), model2.parameters()), lr=initial_lr)

param_num = sum(p.numel() for p in chain(model1.parameters(), model2.parameters()))
print("parameter num: {:.0f}M\n".format(param_num / 1e6))

total_step = len(asl_dataloader)
for epoch in range(total_epoch):
    lr = lr_scheduler(epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    mu = 1 - (1 - 0.99) * (np.cos(np.pi * epoch / total_epoch) + 1) / 2
    gamma = (np.cos(np.pi * epoch / total_epoch) + 1) / 2

    data_iter = iter(asl_dataloader)
    batch_raw = next(data_iter)
    next_batch = [
        batch_raw[0].cuda(non_blocking=True),
        batch_raw[1].cuda(non_blocking=True),
        imm_label.generate_imm_label(batch_raw[1]).cuda(non_blocking=True),
        soft_label_converter.generate_soft_label(batch_raw[1]).cuda(non_blocking=True),
    ]

    start_time = time.time()
    # for i, (data, label) in enumerate(asl_dataloader):
    for i in range(total_step):
        data, label, label2, soft_label = next_batch
        if i + 1 != total_step:
            batch_raw = next(data_iter)
            next_batch = [
                batch_raw[0].cuda(non_blocking=True),
                batch_raw[1].cuda(non_blocking=True),
                imm_label.generate_imm_label(batch_raw[1]).cuda(non_blocking=True),
                soft_label_converter.generate_soft_label(batch_raw[1]).cuda(non_blocking=True),
            ]
        data_time = time.time()

        feature, output = model1(data)
        loss = criterion(output, soft_label)

        output2 = model2(feature)
        loss_imm = criterion(output2, label2)
        loss += gamma * loss_imm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for theta1, theta2 in zip(model1.mlp.parameters(), model2.mlp.parameters()):
                theta1.copy_(mu * theta1 + (1 - mu) * theta2)

        if (i + 1) % 100 == 0:
            step_time = time.time() - start_time
            data_time = data_time - start_time

            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} Time [data {:.1f}ms/step {:.1f}ms] eta {:.1f}min".format(
                    epoch + 1,
                    total_epoch,
                    i + 1,
                    total_step,
                    loss.item(),
                    data_time * 1000,
                    step_time * 1000,
                    total_epoch * total_step * step_time / 60,
                )
            )
            writer.add_scalar("loss/train", loss.item(), epoch * total_step + i + 1)
        start_time = time.time()

    if (epoch + 1) % 10 == 0:
        os.makedirs(result_path, exist_ok=True)
        torch.save(model1.state_dict(), os.path.join(result_path, model_name + f"_1_{epoch + 1:03d}.pt"))
        torch.save(model2.state_dict(), os.path.join(result_path, model_name + f"_2_{epoch + 1:03d}.pt"))

        # eval
        model1.eval()
        avg_loss = 0.0
        avg_accuracy = 0.0
        total_num = 0
        for i, (data, label) in enumerate(test_dataloader):
            data = data.to(device)
            label = label.to(device)
            soft_label = soft_label_converter.generate_soft_label(label).to(device)

            _, output = model1(data)
            loss = criterion(output, soft_label)
            correct_num = (torch.argmax(output, dim=1) == label).sum()

            total_num += batch_size
            avg_loss += loss.item()
            avg_accuracy += correct_num.item()
        avg_loss = avg_loss / total_num
        avg_accuracy = avg_accuracy / total_num
        writer.add_scalar("loss/valid", avg_loss, (epoch + 1) * total_step)
        writer.add_scalar("acc/valid", avg_accuracy, (epoch + 1) * total_step)
        print("Valid epoch {} Loss: {:.4f} Acc: {:.4f}".format(epoch, avg_loss, avg_accuracy))
        model1.train()

writer.flush()
