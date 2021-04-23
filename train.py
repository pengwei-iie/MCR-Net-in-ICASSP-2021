import os
import args
import torch
import random
import pickle
from tqdm import tqdm
from torch import nn, optim

import evaluate
from optimizer import BertAdam
from dataset.dataloader import Dureader
from dataset.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from model_dir.modeling import BertForQuestionAnswering, BertConfig

# 随机种子
random.seed(args.seed)
torch.manual_seed(args.seed)
device = args.device
device_ids = [0]
if len(device_ids) > 0:
    torch.cuda.manual_seed_all(args.seed)

def train():
    # 加载预训练bert
    model = BertForQuestionAnswering.from_pretrained('./roberta_wwm_ext',
                    cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1)))
    # device = args.device
    # model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)  # 声明所有可用设备
        model = model.cuda(device=device_ids[0])  # 模型放在主设备
    elif len(device_ids) == 1:
        model = model.to(device)

    # 准备 optimizer
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=0.1, t_total=args.num_train_optimization_steps)
    # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
    # 准备数据
    data = Dureader()
    train_dataloader, dev_dataloader = data.train_iter, data.dev_iter

    best_loss = 100000.0
    model.train()
    for i in range(args.num_train_epochs):
        main_losses, ide_losses = 0, 0
        for step , batch in enumerate(tqdm(train_dataloader, desc="Epoch")):
            input_ids, input_mask, input_ids_q, input_mask_q, \
            segment_ids, can_answer, start_positions, end_positions = \
                batch.input_ids, batch.input_mask, batch.input_ids_q, batch.input_mask_q,\
                batch.segment_ids, batch.can_answer,\
                batch.start_position, batch.end_position

            flag = torch.ones(4).cuda(device=device_ids[0])

            if len(device_ids) > 1:
                input_ids, input_mask, input_ids_q, input_mask_q, \
                segment_ids, can_answer, start_positions, end_positions = \
                    input_ids.cuda(device=device_ids[0]), input_mask.cuda(device=device_ids[0]), \
                    input_ids_q.cuda(device=device_ids[0]), input_mask_q.cuda(device=device_ids[0]), \
                    segment_ids.cuda(device=device_ids[0]), can_answer.cuda(device=device_ids[0]),\
                    start_positions.cuda(device=device_ids[0]), end_positions.cuda(device=device_ids[0])
            elif len(device_ids) == 1:
                input_ids, input_mask, input_ids_q, input_mask_q, \
                segment_ids, can_answer, start_positions, end_positions = \
                    input_ids.to(device), input_mask.to(device), input_ids_q.to(device), input_mask_q.to(device), \
                    segment_ids.to(device), can_answer.to(device), start_positions.to(device), \
                    end_positions.to(device)
                print("gpu nums is 1.")

            # 计算loss
            loss, main_loss, ide_loss, s, e = model(input_ids, input_ids_q, token_type_ids=segment_ids,
                                                    attention_mask=input_mask, attention_mask_q=input_mask_q,
                                                    can_answer=can_answer, start_positions=start_positions,
                                                    end_positions=end_positions, flag=flag)
            main_losses += main_loss.mean().item()
            ide_losses += ide_loss.mean().item()
            if step % 100 == 0 and step:
                print('After {}, main_losses is {}, ide_losses is {},   ide_losses is dd'.format(step,
                                                                                                 main_losses/step,
                                                                                                 ide_losses/step))
            elif step == 0:
                print('After {}, main_losses is {}, ide_losses is {},   ide_losses is dd'.format(step,
                                                                                                 main_losses,
                                                                                                 ide_losses))
            # loss = loss / args.gradient_accumulation_steps
            # loss.backward()
            # if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            # 更新梯度
            if (step+1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # 验证
            if step % args.log_step == 4:
                eval_loss = evaluate.evaluate(model, dev_dataloader, device_ids)
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    if len(device_ids) > 1:
                        torch.save(model.module.state_dict(), './model_dir/' + "best_model_ablation-mult")
                    if len(device_ids) == 1:
                        torch.save(model.state_dict(), './model_dir/' + "best_model_ablation-mult")
                model.train()

if __name__ == "__main__":
    train()
