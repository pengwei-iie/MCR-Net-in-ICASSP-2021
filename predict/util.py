import json
import args
import torch
import pickle
from tqdm import tqdm
import numpy as np

def creat_examples(filename_1, filename_2, result):         
    count, sum = 0, 0
    examples = []
    lengths = []
    with open(filename_1, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):

            source = json.loads(line.strip())
            lengths.append(len(source['question'].strip()))
            # DESCRIPTION, YES_NO, ENTITY
            if source['question_type'] == args.action:
                source['doc_tokens'] = []
                # 对五篇文档进行处理，保留在source['doc_tokens']
                for doc in source['documents']:
                        ques_len = len(doc['segmented_title']) + 1
                        clean_doc = "".join(doc['segmented_paragraphs'][doc['most_related_para']][ques_len:])
                        if len(clean_doc) > 4:
                            source['doc_tokens'].append( {'doc_tokens': clean_doc} )
                for i in range(len(source['answers'])):
                    count += len(source['answers'][i])
                    sum += 1
                example = ({
                            'id':source['question_id'],
                            'question_text':source['question'].strip(),
                            'question_type': source['question_type'],
                            'doc_tokens':source['doc_tokens'],
                            'answers':source['answers']})
                examples.append(example)
    with open(filename_2, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            source = json.loads(line.strip())
            lengths.append(len(source['question'].strip()))
            # DESCRIPTION, YES_NO, ENTITY
            if source['question_type'] == args.action:
                source['doc_tokens'] = []
                for doc in source['documents']:
                        ques_len = len(doc['segmented_title']) + 1
                        clean_doc = "".join(doc['segmented_paragraphs'][doc['most_related_para']][ques_len:])
                        if len(clean_doc) > 4:
                            source['doc_tokens'].append( {'doc_tokens': clean_doc} )

                if len(source['documents']) == 0:
                    print("error")
                    continue
                for i in range(len(source['answers'])):
                    count += len(source['answers'][i])
                    sum += 1
                example = ({
                            'id':source['question_id'],
                            'question_text':source['question'].strip(),
                            'question_type': source['question_type'],
                            'doc_tokens':source['doc_tokens'],
                            'answers':source['answers'] })
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
    print(count /sum)
    print("{} questions in total".format(len(examples)))
    with open(result, 'wb') as fw:
        pickle.dump(examples, fw)

if __name__ == "__main__":
    creat_examples(filename_1=args.dev_zhidao_input_file,
                   filename_2=args.dev_search_input_file,
                   result=args.predict_example_files)
