import json
from tqdm import tqdm

input_file = "../metric/predicts_baseline_8.json"
output_file = "../metric/submit_baseline.json"
examples = {}
with open(input_file, 'r', encoding='utf-8') as f:
    for line in tqdm(f.readlines()):
        source = json.loads(line.strip())
        ques_id = source['question_id']
        if len(source['answers'][0]) == 0:
            answer = 'no answer'
        elif source['answers'][0] == 'no':
            answer = 'no answer'
        else:
            answer = source['answers'][0]
        examples[ques_id] = answer

with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(examples, json_file, ensure_ascii=False)
# with open(output_file, 'w', encoding="utf-8") as fout:
#     for feature in examples:
#         fout.write(json.dumps(feature, ensure_ascii=False) + '\n')
