import json
import args
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from tokenization import BertTokenizer
import matplotlib.pyplot as plt
from collections import Counter

random.seed(args.seed)

def precision_recall_f1(prediction, ground_truth):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        prediction: prediction string or list to be matched,一篇文档分好词的
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    if not isinstance(prediction, list):
        prediction_tokens = prediction.split()
    else:
        prediction_tokens = prediction
    if not isinstance(ground_truth, list):
        ground_truth_tokens = ground_truth.split()
    else:
        ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)  # 找两个集合里的重复部分，次数是较少的一边
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def recall(prediction, ground_truth):
    """
    This function calculates and returns the recall
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of recall
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[1]


def f1_score(prediction, ground_truth):
    """
    This function calculates and returns the f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of f1
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[2]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        metric_fn: metric function pointer which calculates scores according to corresponding logic.
        prediction: prediction string or list to be matched，一篇文档
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    scores_for_ground_truths = []
    # for ground_truth in ground_truths:
    score = metric_fn(prediction, ground_truths)
    # scores_for_ground_truths.append(score)
    return score


def levenshtein(first, second):
    ''' 编辑距离算法（LevD）
        Args: 两个字符串
        returns: 两个字符串的编辑距离 int
    '''
    if len(first) > len(second):
        first, second = second, first
    if len(first) == 0:
        return len(second)
    if len(second) == 0:
        return len(first)
    first_length = len(first) + 1
    second_length = len(second) + 1
    distance_matrix = [list(range(second_length)) for x in range(first_length)]
    # print distance_matrix
    for i in range(1, first_length):
        for j in range(1, second_length):
            deletion = distance_matrix[i - 1][j] + 1
            insertion = distance_matrix[i][j - 1] + 1
            substitution = distance_matrix[i - 1][j - 1]
            if first[i - 1] != second[j - 1]:
                substitution += 1
            distance_matrix[i][j] = min(insertion, deletion, substitution)
            # print distance_matrix
    return distance_matrix[first_length - 1][second_length - 1]


def read_squad_examples(zhidao_input_file, is_training=True):
    total, error = 0, 0
    examples = []

    with open(zhidao_input_file, 'r', encoding='utf-8') as f:
        lens = []
        lens_ques = []
        for line in tqdm(f.readlines()):

            source = json.loads(line.strip())
            # if (len(source['answer_spans']) == 0):
            #     continue
            # if source['answers'] == []:
            #     continue
            # if (source['match_scores'][0] < 0.8):
            #     continue
            # if (source['answer_spans'][0][1] > args.max_seq_length):
            #     continue

            # if source['documents'] == []:
            #     continue
            # 答案不存在的时候
            if (len(source['answer_spans']) == 0) or source['answers'] == []:
                source['answer_docs'] = [0]
                source['answer_spans'] = [[0, 0]]
                doc_tokens = source['doc_tokens']
                new_doc_tokens = "".join(doc_tokens)
                lens.append(len(new_doc_tokens))
                lens_ques.append(len(source['question'].strip()))
                example = {
                    "qas_id": source['question_id'],
                    "question_text": source['question'].strip(),
                    "doc_tokens": new_doc_tokens.strip(),
                    "can_answer": 0,
                    "start_position": -1,
                    "end_position": -1,
                    "answer": source['answers'].strip()}

                examples.append(example)
                continue

            # if source['match_scores'][0] < 0.8:
            #     continue
            if source['answer_spans'][0][1] > args.max_seq_length:
                continue

            doc_tokens = "".join(source['doc_tokens'])
            lens.append(len(doc_tokens))
            lens_ques.append(len(source['question'].strip()))
            start_id = source['qas'][0]['answers'][0]['answer_start']
            answer = source['qas'][0]['answers'][0]['text']
            end_id = start_id + len(answer)
            # question_type = source['question_type']
            # doc_tokens 分好词的
            # doc_tokens = source['doc_tokens']
            #
            # if start_id >= end_id or end_id > len(doc_tokens) or start_id >= len(doc_tokens):
            #     continue
            #
            # new_doc_tokens = ""
            # new_start_id = 0
            # for idx, token in enumerate(doc_tokens):
            #     if idx == start_id:
            #         new_start_id = len(new_doc_tokens)
            #         break
            #     new_doc_tokens = new_doc_tokens + token
            #
            # new_doc_tokens = "".join(doc_tokens)
            # new_end_id = new_start_id + len(source['fake_answers'][0])
            # # +1的目的是为了这
            # if source['fake_answers'][0] != "".join(new_doc_tokens[new_start_id:new_end_id]):
            #     continue

            if is_training:
                example = {
                    "qas_id": source['question_id'],
                    "question_text": source['question'].strip(),
                    "doc_tokens": doc_tokens.strip(),
                    "can_answer": 1,
                    "start_position": start_id,
                    "end_position": end_id,
                    "answer": answer.strip()}

                examples.append(example)
        # plt.figure("lena")
        # arr = img.flatten()
        # n, bins, patches = plt.hist(probs, bins=50, normed=1, facecolor='green', alpha=0.75)
        # n, bins, patches = plt.hist(probs_can, bins=50, normed=1, facecolor='blue', alpha=0.75)
        # plt.xlabel('score', alpha=0.5)
        # plt.ylabel('frequency', alpha=0.5)
        # plt.legend(['no answer', 'can answer'], loc='upper right')
        # plt.show()
    print("len(examples):", len(examples))
    print("50%, ", np.percentile(lens, 50))
    print("90%, ", np.percentile(lens, 90))
    print("95%, ", np.percentile(lens, 95))
    print("50%, ", np.percentile(lens_ques, 50))
    print("90%, ", np.percentile(lens_ques, 90))
    print("95%, ", np.percentile(lens_ques, 95))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length):
    features = []

    for example in tqdm(examples):
        query_tokens = list(example['question_text'])
        # query_tokens = query_tokens.replace(u"“", u"\"")
        # query_tokens = query_tokens.replace(u"”", u"\"")
        # title = example['title']
        doc_tokens = example['doc_tokens']
        doc_tokens = doc_tokens.replace(u"“", u"\"")
        doc_tokens = doc_tokens.replace(u"”", u"\"")
        start_position = example['start_position']
        end_position = example['end_position']
        can_answer = example['can_answer']

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tokens = []
        tokens_q = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)
        if start_position != -1:
            start_position = start_position + 1
            end_position = end_position + 1

        for token in query_tokens:
            tokens_q.append(token)
            tokens.append(token)
            segment_ids.append(0)
            if start_position != -1:
                start_position = start_position + 1
                end_position = end_position + 1

        tokens.append("[SEP]")
        segment_ids.append(0)
        if start_position != -1:
            start_position = start_position + 1
            end_position = end_position + 1

        for i in doc_tokens:
            tokens.append(i)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        if end_position >= max_seq_length:
            continue

        if len(tokens) > max_seq_length:
            tokens[max_seq_length - 1] = "[SEP]"
            input_ids = tokenizer.convert_tokens_to_ids(tokens[:max_seq_length])  ## !!! SEP
            segment_ids = segment_ids[:max_seq_length]
        else:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_ids_q = tokenizer.convert_tokens_to_ids(tokens_q)

        input_mask = [1] * len(input_ids)
        input_mask_q = [1] * len(input_ids_q)
        assert len(input_ids) == len(segment_ids)
        assert len(input_ids_q) == len(input_mask_q)

        features.append(
            {"input_ids": input_ids,
             "input_ids_q": input_ids_q,
             # "tokens": tokens,
             # "title": title,
             "input_mask": input_mask,
             "input_mask_q": input_mask_q,
             "segment_ids": segment_ids,
             "can_answer": can_answer,
             "start_position": start_position,
             "end_position": end_position})
    print("len(features):", len(features))
    with open("./train_2021.json", 'w', encoding="utf-8") as fout:
        for feature in features:
            fout.write(json.dumps(feature, ensure_ascii=False) + '\n')

    return features


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('../chinese_roberta_wwm_ext', do_lower_case=True)
    # 生成训练数据， train.data
    # examples = read_squad_examples(zhidao_input_file=args.zhidao_input_file,
    #                                search_input_file=args.search_input_file)
    # features = convert_examples_to_features(examples=examples, tokenizer=tokenizer,
    #                                         max_seq_length=args.max_seq_length, max_query_length=args.max_query_length)

    # 生成验证数据， dev.data。记得注释掉生成训练数据的代码，并在196行将train.data改为dev.data
    examples = read_squad_examples(zhidao_input_file=args.train_2021)
    features = convert_examples_to_features(examples=examples, tokenizer=tokenizer,
                                            max_seq_length=args.max_seq_length, max_query_length=args.max_query_length)
