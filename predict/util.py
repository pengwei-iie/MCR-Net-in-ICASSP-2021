import json
import args
import torch
import pickle
from tqdm import tqdm
import numpy as np


def creat_examples(filename_1, result):
    count, sum = 0, 0
    examples = []
    lengths = []
    samples = json.load(open(filename_1, 'r', encoding='utf-8'))['data'][0]['paragraphs']
    for source in samples:
        # DESCRIPTION, YES_NO, ENTITY

        if len(source['qas']) == 0:
            continue
        for i in range(len(source['qas'])):
            clean_doc = source['context']
            source['doc_tokens'] = []
            source['doc_tokens'].append({'doc_tokens': clean_doc})
            source['question_id'] = source['qas'][i]['id']
            source['question'] = source['qas'][i]['question']

            example = ({
                'id': source['question_id'],
                'question_text': source['question'].strip(),
                'doc_tokens': source['doc_tokens'],
                'answers': 'test'})
            examples.append(example)

    # print the distribution of the question
    # y = sorted(lengths)
    # print(y)
    # # 统计出现的元素有哪些
    # unique_data = np.unique(y)
    # print(unique_data)
    #
    # # 统计某个元素出现的次数
    # resdata = []
    # for ii in unique_data:
    #     resdata.append(y.count(ii))
    # print(resdata)
    # print(count /sum)
    print("{} questions in total".format(len(examples)))
    with open(result, 'wb') as fw:
        pickle.dump(examples, fw)


if __name__ == "__main__":
    predict_file = '../dataset_checklist/test1.json'
    creat_examples(filename_1=predict_file,
                   result=args.predict_example_files)
