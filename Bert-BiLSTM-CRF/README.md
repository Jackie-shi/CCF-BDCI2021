#### 

##### bert + bilstm + crf
* model_name：模型名称
* epochs：迭代epoch的数量
* checkpoint_every：间隔多少步保存一次模型
* eval_every：间隔多少步验证一次模型
* learning_rate：学习速率
* sequence_length：序列长度，单GPU时不要超过128
* batch_size：训练时batch_size
* batch_test_size：测试时batch_size
* ner_layers：lstm中的隐层大小
* ner_hidden_sizes：bilstm-ner中的全连接层中的隐层大小
* keep_prob：bilstm-ner中全连接层中的dropout比例
* num_classes：NER标签数量
* warmup_rate：训练时的预热比例
* output_path：输出文件夹，用来存储label_to_index等文件
* bert_model_path：预训练模型文件夹路径
* train_data：训练数据路径
* eval_data：验证数据路径
* test_data：测试集数据路径
* ckpt_model_path：checkpoint模型文件保存路径

<img width="3626" alt="Bert-BiLSTM-CRF 代码分析" src="https://user-images.githubusercontent.com/75195605/135010580-e73d9411-f245-4ec8-907f-d6d10ce64f34.png">
