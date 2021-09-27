import json
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import tensorflow as tf
from model import BertNer
from bert import tokenization
from metrics import get_chunk
import io
import argparse
import pandas as pd

class Predictor(object):
    def __init__(self, config):
        self.model = None
        self.config = config
        self.batch_size = config["batch_test_size"]
        self.output_path = config["output_path"]
        self.vocab_path = os.path.join(config["bert_model_path"], "vocab.txt")
        self.label_to_index = self.load_vocab()
        self.index_to_label = {value: key for key, value in self.label_to_index.items()}
        self.word_vectors = None
        self.sequence_length = self.config["sequence_length"]

        # 创建模型
        self.create_model()
        # 加载计算图
        self.load_graph()

    def load_vocab(self):
        # 将词汇-索引映射表加载出来

        with open(os.path.join(self.output_path, "label_to_index.json"), "r") as f:
            label_to_index = json.load(f)

        return label_to_index

    @staticmethod
    def read_data(file_path):
        """
        读取数据
        :param file_path:
        :return: 返回分词后的文本内容和标签，inputs = [], labels = []
        """
        text = pd.read_csv(file_path, usecols=["text"])
        label = pd.read_csv(file_path, usecols=["BIO_anno"])
        inputs = []
        labels = []
        text = text.values.tolist()
        label = label.values.tolist()
        for i in range(len(text)):
            inputs.append([one for one in text[i][0].strip()])
            labels.append(label[i][0].strip().split(" "))
        return inputs, labels

    def padding(self, input_ids, input_masks, segment_ids):
        """
        对序列进行补全
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :return:
        """
        pad_input_ids, pad_input_masks, pad_segment_ids, sequence_len = [], [], [], []
        for input_id, input_mask, segment_id in zip(input_ids, input_masks, segment_ids):
            if len(input_id) < self.sequence_length:
                pad_input_ids.append(input_id + [0] * (self.sequence_length - len(input_id)))
                pad_input_masks.append(input_mask + [0] * (self.sequence_length - len(input_mask)))
                pad_segment_ids.append(segment_id + [0] * (self.sequence_length - len(segment_id)))
                sequence_len.append(len(input_id))
            else:
                pad_input_ids.append(input_id[:self.sequence_length])
                pad_input_masks.append(input_mask[:self.sequence_length])
                pad_segment_ids.append(segment_id[:self.sequence_length])
                sequence_len.append(self.sequence_length)

        return pad_input_ids, pad_input_masks, pad_segment_ids, sequence_len
    
    def sentence_to_idx(self, inputs):
        """
        将输入转化为索引表示
        :param inputs: 输入
        :return:
        """
        tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_path, do_lower_case=True)
        input_ids = []
        input_masks = []
        segment_ids = []

        for text in inputs:
            tokens = []
            for token in text:
                token = tokenizer.tokenize(token)
                tokens.extend(token)

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_id = tokenizer.convert_tokens_to_ids(tokens)

            input_ids.append(input_id)
            input_masks.append([1] * len(input_id))
            segment_ids.append([0] * len(input_id))
        
        input_ids, input_masks, segment_ids, sequence_len = self.padding(input_ids, input_masks, segment_ids)

        return input_ids, input_masks, segment_ids, sequence_len
    
    def load_graph(self):
        """
        加载计算图
        :return:
        """
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.config["ckpt_model_path"])
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(self.config["ckpt_model_path"]))

    def create_model(self):
        """
                根据config文件选择对应的模型，并初始化
                :return:
                """
        self.model = BertNer(config=self.config, is_training=False)

    def next_batch(self, input_ids, input_masks, segment_ids, sequence_len):
        """
        生成batch数据
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :param sequence_len:
        :return:
        """
        z = list(zip(input_ids, input_masks, segment_ids, sequence_len))
        input_ids, input_masks, segment_ids, sequence_len = zip(*z)

        num_batches = len(input_ids) // self.batch_size

        for i in range(num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            batch_input_ids = input_ids[start: end]
            batch_input_masks = input_masks[start: end]
            batch_segment_ids = segment_ids[start: end]
            batch_sequence_len = sequence_len[start: end]

            yield dict(input_ids=batch_input_ids,
                       input_masks=batch_input_masks,
                       segment_ids=batch_segment_ids,
                       sequence_len=batch_sequence_len)
    
    def predict(self, batch):
        """
        给定分词后的句子，预测其分类结果
        :param text:
        :return:
        """
        prediction = self.model.infer(self.sess, batch)
        # print(prediction)
        prediction_tag = [self.index_to_label[i] for i in list(prediction)]
        chunks = get_chunk(prediction, self.label_to_index)
        # print(chunks)
        return prediction_tag


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="config path of model")
    args = parser.parse_args()
    with open(args.config_path, "r") as fr:
        config = json.load(fr)
    predictorer = Predictor(config)
    text, _ = predictorer.read_data(config["test_data"])
    input_ids, input_masks, segment_ids, sequence_len = predictorer.sentence_to_idx(text)
    pre_lable = []
    for test_batch in predictorer.next_batch(input_ids, input_masks, segment_ids, sequence_len):
        prediction_tag = predictorer.predict(test_batch)
        # true_label += list(test_batch["label_ids"])
        pre_lable.append(prediction_tag)
        # print(pre_lable)
        # acc = accuracy_score(list(test_batch["label_ids"]), list(predictions))
        # eval_accs.append(acc)
    print(pre_lable)
    # output_file = os.path.join(config["output_path"], "result.txt")
    # fp = open(output_file, 'w')

    # for t, p in zip(true_label, pre_lable):
    #     fp.write(str(t) + ' ' + str(p) + '\n')
