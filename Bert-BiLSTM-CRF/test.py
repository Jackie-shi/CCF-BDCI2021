import json
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import tensorflow as tf
from model import BertNer
from bert import tokenization
from metrics import get_chunk, gen_metrics, mean 
import io
import argparse
import pandas as pd

class Test(object):
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

    @staticmethod
    def trans_label_to_index(labels, label_to_index):
        """
        将标签也转换成数字表示
        :param labels: 标签
        :param label_to_index: 标签-索引映射表
        :return:
        """
        labels_ids = [[label_to_index[item] for item in label] for label in labels]
        return labels_ids
    
    def padding(self, input_ids, input_masks, segment_ids, label_ids, label_to_index):
        """
        对序列进行补全
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :param label_ids
        :param label_to_index
        :return:
        """
        pad_input_ids, pad_input_masks, pad_segment_ids, pad_label_ids, sequence_len = [], [], [], [], []
        for input_id, input_mask, segment_id, label_id in zip(input_ids, input_masks, segment_ids, label_ids):
            if len(input_id) < self.sequence_length:
                pad_input_ids.append(input_id + [0] * (self.sequence_length - len(input_id)))
                pad_input_masks.append(input_mask + [0] * (self.sequence_length - len(input_mask)))
                pad_segment_ids.append(segment_id + [0] * (self.sequence_length - len(segment_id)))
                pad_label_ids.append(label_id + [label_to_index["O"]] * (self.sequence_length - len(label_id)))
                sequence_len.append(len(input_id))
            else:
                pad_input_ids.append(input_id[:self.sequence_length])
                pad_input_masks.append(input_mask[:self.sequence_length])
                pad_segment_ids.append(segment_id[:self.sequence_length])
                pad_label_ids.append(label_id[:self.sequence_length])
                sequence_len.append(self.sequence_length)

        return pad_input_ids, pad_input_masks, pad_segment_ids, pad_label_ids, sequence_len
    
    def sentence_to_idx(self, inputs, labels):
        """
        将输入转化为索引表示
        :param inputs: 输入
        :return:
        """
        tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_path, do_lower_case=True)
        input_ids = []
        input_masks = []
        segment_ids = []
        new_labels = []

        for text, label in zip(inputs, labels):

            tokens = []
            new_label = []
            for token, tag in zip(text, label):
                token = tokenizer.tokenize(token)
                tokens.extend(token)
                new_label.extend([tag] * len(token))

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_id = tokenizer.convert_tokens_to_ids(tokens)

            label = ["O"] + label + ["O"]

            input_ids.append(input_id)
            input_masks.append([1] * len(input_id))
            segment_ids.append([0] * len(input_id))
            new_labels.append(label)
        
        labels_ids = self.trans_label_to_index(new_labels, self.label_to_index)

        input_ids, input_masks, segment_ids, labels_ids, sequence_len = self.padding(input_ids,
                                                                                      input_masks,
                                                                                      segment_ids,
                                                                                      labels_ids,
                                                                                      self.label_to_index)

        return input_ids, input_masks, segment_ids, labels_ids, sequence_len
    
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

    def next_batch(self, input_ids, input_masks, segment_ids, label_ids, sequence_len):
        """
        生成batch数据
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :param label_ids:
        :param sequence_len:
        :return:
        """
        z = list(zip(input_ids, input_masks, segment_ids, label_ids, sequence_len))
        input_ids, input_masks, segment_ids, label_ids, sequence_len = zip(*z)

        num_batches = len(input_ids) // self.batch_size

        for i in range(num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            batch_input_ids = input_ids[start: end]
            batch_input_masks = input_masks[start: end]
            batch_segment_ids = segment_ids[start: end]
            batch_label_ids = label_ids[start: end]
            batch_sequence_len = sequence_len[start: end]

            yield dict(input_ids=batch_input_ids,
                       input_masks=batch_input_masks,
                       segment_ids=batch_segment_ids,
                       label_ids=batch_label_ids,
                       sequence_len=batch_sequence_len)
    
    def predict(self, batch):
        """
        给定分词后的句子，预测其分类结果
        :param text:
        :return:
        """
        true_label, prediction = self.model.infer_1(self.sess, batch)
        prediction_tag = [self.index_to_label[i] for i in list(prediction)]
        chunks = get_chunk(prediction, self.label_to_index)
        # print(chunks)
        return prediction_tag, true_label, prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="config path of model")
    args = parser.parse_args()
    with open(args.config_path, "r") as fr:
        config = json.load(fr)
    predictorer = Test(config)
    text, labels = predictorer.read_data(config["test_data"])
    input_ids, input_masks, segment_ids, labels_ids, sequence_len = predictorer.sentence_to_idx(text, labels)
    test_recalls, test_precisions, test_f1s = [], [], []
    for test_batch in predictorer.next_batch(input_ids, input_masks, segment_ids, labels_ids, sequence_len):
        prediction_tag, true_y, predictions = predictorer.predict(test_batch)
        f1, precision, recall = gen_metrics(pred_y=predictions, true_y=true_y, label_to_index=predictorer.label_to_index)
        test_recalls.append(recall)
        test_precisions.append(precision)
        test_f1s.append(f1)
    print("test:  recall: {}, precision: {}, f1: {}".format(mean(test_recalls), mean(test_precisions), mean(test_f1s)))