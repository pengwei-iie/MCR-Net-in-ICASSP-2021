import torch

seed = 42
device = torch.device("cuda", 0)
test_lines = 3000  # 多少条训练数据，即：len(features), 记得修改 !!!!!!!!!!

search_input_file = "../data/extracted/trainset/search.train.json"
zhidao_input_file = "../data/extracted/trainset/zhidao.train.json"
dev_zhidao_input_file = "../data/extracted/devset/zhidao.dev.json"
dev_search_input_file = "../data/extracted/devset/search.dev.json"
'''
50%,  157.0
90%,  445.0
95%,  567.1500000000001
50%,  10.0
90%,  14.0
95%,  16.0
'''
max_seq_length = 460
max_query_length = 16
# max_seq_length = 512
# max_query_length = 60

# pretrained_file = "./chinese_roberta_wwm_l"
output_dir = "./model_dir_"
predict_example_files='predict_test1.data'

max_para_num=5  # 选择几篇文档进行预测
learning_rate = 2e-5
batch_size = 4
num_train_epochs = 8
gradient_accumulation_steps = 8   # 梯度累积
num_train_optimization_steps = int(test_lines / gradient_accumulation_steps / batch_size) * num_train_epochs
log_step = int(test_lines / batch_size / 4)  # 每个epoch验证几次，默认4次
