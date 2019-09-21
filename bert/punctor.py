# -*- coding: utf-8 -*-
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
SPACE='_SPACE'
PUNCTUATION_VOCABULARY = [SPACE, ",COMMA", ".PERIOD", "?QUESTIONMARK"]
tag_dict = {"_SPACE": 0, ",COMMA": 1, ".PERIOD": 2, "?QUESTIONMARK": 3}
PUNCTUATION_MAPPING = {"!EXCLAMATIONMARK": ".PERIOD", ":COLON": ",COMMA", ";SEMICOLON": ".PERIOD",
                       "-DASH": ",COMMA"}
EOS_TOKENS = {".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK"}
END = "</S>"
import json
import numpy as np
import pickle
import codecs
import collections
import csv
import os
import itertools
from numpy import nan
from bert import modeling,optimization,tokenization,tf_metrics
import tensorflow as tf
from bert.lstm_crf_layer import BLSTM_CRF
from tensorflow.contrib.layers.python.layers import initializers

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
flags = tf.flags

FLAGS = flags.FLAGS


## Required parameters
flags.DEFINE_string(
    "data_dir", '/data/dh/neural_sequence_labeling-master/data/dataset/lrec/',
    "The input data dir. Should contain the json files (or other data files) "
    "for the task.")
flags.DEFINE_string('data_config_path', '/data/dh/neural_sequence_labeling-master/bert/outputdata.conf',
                    'data config file, which save train and dev config')
flags.DEFINE_string(
    "test_data_dir", '/data/dh/neural_sequence_labeling-master/data/raw/LREC_converted/',
    "The input data dir. Should contain the .txt files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", '/data/dh/neural_sequence_labeling-master/uncased_L-24_H-1024_A-16/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name",'Punctor' , "The name of the task to train.")

flags.DEFINE_string("vocab_file", '/data/dh/neural_sequence_labeling-master/uncased_L-24_H-1024_A-16/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", '/data/dh/neural_sequence_labeling-master/bert/output',
    "The output directory where the model checkpoints will be written.")

## Other parameters
''
flags.DEFINE_string(
    "init_checkpoint", '/data/dh/neural_sequence_labeling-master/uncased_L-24_H-1024_A-16/bert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 199,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
flags.DEFINE_boolean('clean', True, 'remove the files which created by last training')
flags.DEFINE_bool("do_train",False, "Whether to run training.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", True, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 4, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 4, "Total batch size for eval.")


flags.DEFINE_integer("predict_batch_size", 4, "Total batch size for predict.")
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")
flags.DEFINE_float('droupout_rate', 0.5, 'Dropout rate')
flags.DEFINE_float('clip', 5, 'Gradient clip')
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")



tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

# lstm parame
flags.DEFINE_integer('lstm_size', 128, 'size of lstm units')
flags.DEFINE_integer('num_layers', 1, 'number of rnn layers, default is 1')
flags.DEFINE_string('cell', 'lstm', 'which rnn cell used')

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""

        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            lines = []
            print('length of tags:',len(data[0]['tags']))
            # length of  tags: 199
            # length of words: 199
            print('length of words:', len(data[0]['words']))
            for item in data:

                l = ' '.join([label for label in item['tags'] if len(label) > 0])
                w = ' '.join([word for word in item['words']  if len(word) > 0])
                lines.append([l, w])
        return lines

class PunctorProcessor(DataProcessor):
    """Processor for the TEA data set."""

    def __init__(self):
        self.language = "en"
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "bert_train.json")), "train"
        )

    def get_dev_examples(self, data_dir):

        return self._create_example(
            self._read_data(os.path.join(data_dir, "bert_dev.json")), "dev"
        )
    def get_test_examples(self,data_dir):#直接按max_seq_length来取，每个example 没有重复的部分

        input_file = os.path.join(data_dir, 'asr.txt')
        index = 0
        lines = []

        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            text = f.read().split()

            text = [w for w in text if w not in tag_dict and w not in PUNCTUATION_MAPPING]
            while index+FLAGS.max_seq_length<len(text) :

                subseq = text[index: index + FLAGS.max_seq_length]

                w = ' '.join([word for word in subseq if len(word) > 0])
                lines.append([w])
                index = index+FLAGS.max_seq_length
            remain_seq = text[index:]


            w = ' '.join([word for word in remain_seq if len(word) > 0])
            lines.append([w])
            return self._create_example(lines, "test")
    def get_test_examples_last_eos(self,data_dir):#每个example的开头都从上一个EOS开始算，所以不同example有重复的词

        input_file = os.path.join(data_dir, 'asr.txt')
        index = 0
        lines = []

        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            text = f.read().split()
            labels = []
            count_punct = 0
            for i in range(0, len(text)):
                if text[i] not in tag_dict and text[i] not in PUNCTUATION_MAPPING:
                    if i + 1 < len(text) and (text[i + 1] not in tag_dict and text[i + 1] not in PUNCTUATION_MAPPING):
                        labels.append('_SPACE')

                    elif i + 1 < len(text) and text[i + 1] in tag_dict:
                        labels.append(text[i + 1])
                        count_punct += 1
                        i += 1
                    else:
                        break
            text = [w for w in text if w not in tag_dict and w not in PUNCTUATION_MAPPING] + [END]

            while True:
                subseq = text[index: index + 199]
                sublabel = labels[index:index + 199]
                l = ' '.join([label for label in sublabel if len(label) > 0])
                w = ' '.join([word for word in subseq if len(word) > 0])
                lines.append([l, w])
                last_eos_idx = 0
                punctuations = []
                for i in sublabel:
                    punctuations.append(i)
                    if i in EOS_TOKENS:
                        last_eos_idx = len(punctuations)
                if subseq[-1] == END:
                    step = len(subseq) - 1
                elif last_eos_idx != 0:
                    step = last_eos_idx
                else:
                    step = len(subseq) - 1
                if subseq[-1] == END:
                    break
                index += step
            lines[-1][1] = lines[-1][1][:-5]
        return self._create_example(lines, "test")
    def get_labels(self):
        """See base class."""
        return ["_SPACE", ",COMMA", ".PERIOD", "?QUESTIONMARK"]
    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type=='test':
                text = tokenization.convert_to_unicode(line[0])
                label = None
            else:
                text = tokenization.convert_to_unicode(line[1])
                label = tokenization.convert_to_unicode(line[0])

            if i == 0:
                print(label)
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples
def write_tokens(tokens, mode):
    """
    将序列解析结果写入到文件中
    只在mode=test的时候启用
    :param tokens:
    :param mode:
    :return:
    """
    if mode == "test":
        path = os.path.join(FLAGS.output_dir, "token_" + mode + ".txt")
        wf = codecs.open(path, 'a', encoding='utf-8')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()
def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param mode:
    :return:
    """
    if mode=='test':
        label_map = {}
        # 1表示从1开始对label进行index化
        for (i, label) in enumerate(label_list):
            label_map[label] = i
        # 保存label->index 的map
        with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)
        tokens = example.text.split(' ')


        # for i, word in enumerate(textlist):
        #     # 分词，如果是中文，就是分字
        #     token = tokenizer.tokenize(word)
        #     tokens.extend(token)
        #     label_1 = labellist[i]
        #     for m in range(len(token)):
        #         if m == 0:
        #             labels.append(label_1)
        #         else:  # 一般不会出现else
        #             labels.append("X")
        # tokens = tokenizer.tokenize(example.text)
        # 序列截断
        # if len(tokens) >= max_seq_length - 1:
        #     tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        #     labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        # ntokens.append("[CLS]")  # 句子开始设置CLS 标志
        # segment_ids.append(0)
        # append("O") or append("[CLS]") not sure!
        # label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            # if mode=='test':
            #     print(tokens)
            #     print(labels)
            #     print(labels[i])
            #     print(label_map)
            #     print(label_map[labels[i]])
            label_ids.append(0)
        # ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
        # segment_ids.append(0)
        # append("O") or append("[SEP]") not sure!
        # label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
        input_mask = [1] * len(input_ids)
        # label_mask = [1] * len(input_ids)
        # padding, 使用
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            label_ids.append(0)
            ntokens.append("**NULL**")
            # label_mask.append(0)
        # print(len(input_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        # assert len(label_mask) == max_seq_length
    else:
        label_map = {}
        # 1表示从1开始对label进行index化
        for (i, label) in enumerate(label_list):
            label_map[label] = i
        # 保存label->index 的map
        with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)
        tokens = example.text.split(' ')
        labels = example.label.split(' ')

        # for i, word in enumerate(textlist):
        #     # 分词，如果是中文，就是分字
        #     token = tokenizer.tokenize(word)
        #     tokens.extend(token)
        #     label_1 = labellist[i]
        #     for m in range(len(token)):
        #         if m == 0:
        #             labels.append(label_1)
        #         else:  # 一般不会出现else
        #             labels.append("X")
        # tokens = tokenizer.tokenize(example.text)
        # 序列截断
        # if len(tokens) >= max_seq_length - 1:
        #     tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        #     labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        #ntokens.append("[CLS]")  # 句子开始设置CLS 标志
        #segment_ids.append(0)
        # append("O") or append("[CLS]") not sure!
        #label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            # if mode=='test':
            #     print(tokens)
            #     print(labels)
            #     print(labels[i])
            #     print(label_map)
            #     print(label_map[labels[i]])
            label_ids.append(label_map[labels[i]])
       # ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
        #segment_ids.append(0)
        # append("O") or append("[SEP]") not sure!
        #label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
        input_mask = [1] * len(input_ids)
        # label_mask = [1] * len(input_ids)
        # padding, 使用
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            label_ids.append(0)
            ntokens.append("**NULL**")
            # label_mask.append(0)
        # print(len(input_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        # assert len(label_mask) == max_seq_length

    # 打印部分样本数据信息
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    # mode='test'的时候才有效
    write_tokens(ntokens, mode)
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file,mode=None):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if example.text=='':
            continue
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer,mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
def create_model(bert_config, is_training, input_ids, input_mask,#这个是bert之后加了lstm和crf的模型 不建议使用
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    """
    创建模型
    :param bert_config: bert 配置
    :param is_training:
    :param input_ids: 数据的idx 表示
    :param input_mask:
    :param segment_ids:
    :param labels: 标签的idx 表示
    :param num_labels: 类别数量
    :param use_one_hot_embeddings:
    :return:
    """
    # 使用数据加载BertModel,获取对应的字embedding
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )
    # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value

    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度

    blstm_crf = BLSTM_CRF(embedded_chars=embedding, hidden_unit=FLAGS.lstm_size, cell_type=FLAGS.cell, num_layers=FLAGS.num_layers,
                          droupout_rate=FLAGS.droupout_rate, initializers=initializers, num_labels=num_labels,
                          seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)
    rst = blstm_crf.add_blstm_crf_layer()
    return rst

def softmax_create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    """
    创建X模型
    :param bert_config: bert 配置
    :param is_training:
    :param input_ids: 数据的idx 表示
    :param input_mask:
    :param segment_ids:
    :param labels: 标签的idx 表示
    :param num_labels: 类别数量
    :param use_one_hot_embeddings:
    :return:
    """
    # 使用数据加载BertModel,获取对应的字embedding
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )
    # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value

    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度
    logits = tf.layers.dense(embedding,units=num_labels,use_bias=True)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)



def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,#这个是bert之后加了lstm和crf的模型 不建议使用
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """
    构建模型
    :param bert_config:
    :param num_labels:
    :param init_checkpoint:
    :param learning_rate:
    :param num_train_steps:
    :param num_warmup_steps:
    :param use_tpu:
    :param use_one_hot_embeddings:
    :return:
    """

    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        print('shape of input_ids', input_ids.shape)
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
        (total_loss, logits, trans, pred_ids) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        scaffold_fn = None
        # 加载BERT模型
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")

        # 打印加载模型的参数
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)  # 钩子，这里用来将BERT中的参数作为我们模型的初始值
        elif mode == tf.estimator.ModeKeys.EVAL:
            # 针对NER ,进行了修改
            # def metric_fn(label_ids, logits, trans):
            #     # 首先对结果进行维特比解码
            #     # crf 解码
            #     weight = tf.sequence_mask(FLAGS.max_seq_length)
            #     precision = tf_metrics.precision(label_ids, pred_ids, num_labels, [0, 1, 2, 3], weight)
            #     recall = tf_metrics.recall(label_ids, pred_ids, num_labels, [0, 1, 2, 3], weight)
            #     f = tf_metrics.f1(label_ids, pred_ids, num_labels, [0, 1, 2, 3], weight)
            #
            #     return {
            #         "eval_precision": precision,
            #         "eval_recall": recall,
            #         "eval_f": f,
            #         #"eval_loss": loss,
            #     }
            #
            # eval_metrics = (metric_fn, [label_ids, logits, trans])
            # # eval_metrics = (metric_fn, [label_ids, logits])
            # output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            #     mode=mode,
            #     loss=total_loss,
            #     eval_metrics=eval_metrics,
            #     scaffold_fn=scaffold_fn)  #

            def metric_fn(total_loss, label_ids, pred_ids):

                accuracy = tf.metrics.accuracy(label_ids, pred_ids)
                #loss = tf.metrics.mean(per_example_loss)
                return {
                    "eval_accuracy": accuracy,
                    "eval_total_loss": total_loss,
                }

            eval_metrics = (metric_fn, [total_loss, label_ids, pred_ids])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)



        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=pred_ids,
                scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn
def softmax_model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
      """Returns `model_fn` closure for TPUEstimator."""

      def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
          tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = softmax_create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
          (assignment_map, initialized_variable_names
          ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
          if use_tpu:

            def tpu_scaffold():
              tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
              return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
          else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
          init_string = ""
          if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
          tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                          init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

          train_op = optimization.create_optimizer(
              total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

          output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=total_loss,
              train_op=train_op,
              scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

          def metric_fn(per_example_loss, label_ids, logits):
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            accuracy = tf.metrics.accuracy(label_ids, predictions)
            loss = tf.metrics.mean(per_example_loss)
            return {
                "eval_accuracy": accuracy,
                "eval_loss": loss,
            }

          eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
          output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=total_loss,
              eval_metrics=eval_metrics,
              scaffold_fn=scaffold_fn)
        else:
          output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode, predictions=probabilities, scaffold_fn=scaffold_fn)
        return output_spec

      return model_fn

# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features

def main():

    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        'punctor': PunctorProcessor,
    }

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    if FLAGS.clean and FLAGS.do_train:
        if os.path.exists(FLAGS.output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)
            try:
                del_file(FLAGS.output_dir)
            except Exception as e:
                print(e)
                print('please remove the files of output dir and data.conf')
                exit(-1)
        if os.path.exists(FLAGS.data_config_path):
            try:
                os.remove(FLAGS.data_config_path)
            except Exception as e:
                print(e)
                print('please remove the files of output dir and data.conf')
                exit(-1)
   # tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if os.path.exists(FLAGS.data_config_path):
        with codecs.open(FLAGS.data_config_path) as fd:
            data_config = json.load(fd)
    else:
        data_config = {}
    if FLAGS.do_train:
        # 加载训练数据
        if len(data_config) == 0:
            train_examples = processor.get_train_examples(FLAGS.data_dir)
            num_train_steps = int(
                len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
            num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

            data_config['num_train_steps'] = num_train_steps
            data_config['num_warmup_steps'] = num_warmup_steps
            data_config['num_train_size'] = len(train_examples)
        else:
            num_train_steps = int(data_config['num_train_steps'])
            num_warmup_steps = int(data_config['num_warmup_steps'])
    # 返回的model_dn 是一个函数，其定义了模型，训练，评测方法，并且使用钩子参数，加载了BERT模型的参数进行了自己模型的参数初始化过程
    # tf 新的架构方法，通过定义model_fn 函数，定义模型，然后通过EstimatorAPI进行模型的其他工作，Es就可以控制模型的训练，预测，评估工作等。
    model_fn = softmax_model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        if data_config.get('eval.tf_record_path', '') == '':
            eval_examples = processor.get_dev_examples(FLAGS.data_dir)
            eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
            filed_based_convert_examples_to_features(
                eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
            data_config['eval.tf_record_path'] = eval_file
            data_config['num_eval_size'] = len(eval_examples)
        else:
            eval_file = data_config['eval.tf_record_path']
        # 打印验证集数据信息
        num_eval_size = data_config.get('num_eval_size', 0)
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", num_eval_size)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(num_eval_size / FLAGS.eval_batch_size)
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with codecs.open(output_eval_file, "w", encoding='utf-8') as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    # 保存数据的配置文件，避免在以后的训练过程中多次读取训练以及测试数据集，消耗时间
    if not os.path.exists(FLAGS.data_config_path):
        with codecs.open(FLAGS.data_config_path, 'a', encoding='utf-8') as fd:
            json.dump(data_config, fd)


    # Metric code
    # pred_result = estimator.predict(input_fn=eval_input_fn)
    # get_eval(pred_result, real_labels, label_list, FLAGS.max_seq_length)
    if FLAGS.do_predict:
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        if os.path.exists(token_path):
            os.remove(token_path)

        with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        predict_examples = processor.get_test_examples(FLAGS.test_data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, label_list,FLAGS.max_seq_length, tokenizer,predict_file, mode="test")

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        # predicted_result = estimator.evaluate(input_fn=predict_input_fn)
        # output_eval_file = os.path.join(FLAGS.output_dir, "predicted_results.txt")
        # with codecs.open(output_eval_file, "w", encoding='utf-8') as writer:
        #     tf.logging.info("***** Predict results *****")
        #     for key in sorted(predicted_result.keys()):
        #         tf.logging.info("  %s = %s", key, str(predicted_result[key]))
        #         writer.write("%s = %s\n" % (key, str(predicted_result[key])))

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")
        def result_to_pair(writer):
            for predict_line, prediction in zip(predict_examples, result):
                prediction = np.argmax(prediction,axis=-1)
                line = []
                line_token = str(predict_line.text).split(' ')

                if len(line_token) != len(prediction):
                    tf.logging.info(predict_line.text)
                    tf.logging.info(predict_line.label)
                print(line_token)
                print(prediction)
                for word,label in zip(line_token,prediction):
                    curr_labels = id2label[label]
                    if curr_labels=="_SPACE":
                        line.append(word)
                    else:
                        line.append(word)
                        line.append(curr_labels)
                line = ' '+' '.join(line)
                print(line)

                writer.write(line)
        def result_to_pair_last_eos(writer):#这个方法对应于get_test_examples_last_eos

            for predict_line, prediction in zip(predict_examples, result):
                prediction = np.argmax(prediction,axis=-1)
                line = []
                line_token = str(predict_line.text).split(' ')
                label_token = str(predict_line.label).split(' ')
                if len(line_token) != len(prediction):
                    tf.logging.info(predict_line.text)
                    tf.logging.info(predict_line.label)
                print(line_token)
                print(prediction)
                last_eos_id = 0
                for i in range(0, len(label_token)):
                    if label_token[i] == '.PERIOD' or label_token[i] == '?QUESTIONMARK':
                        last_eos_id = i
                if last_eos_id + 1 < len(line_token):
                    line_token = line_token[:last_eos_id + 1]
                for word,label in zip(line_token,prediction):

                    curr_labels = id2label[label]
                    if curr_labels=="_SPACE":
                        line.append(word)
                    else:
                        line.append(word)
                        line.append(curr_labels)
                line = ' '+' '.join(line)
                print(line)

                writer.write(line)

        with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:
            result_to_pair(writer)
        target_path =  '/data/dh/neural_sequence_labeling-master/data/raw/LREC_converted/asr.txt'
        predict_path = '/data/dh/neural_sequence_labeling-master/bert/output/label_test.txt'
        out_str, f1, err, ser = compute_score(target_path,predict_path)
        tf.logging.info("\nEvaluate on {}:\n{}\n".format('asr', out_str))


def compute_score(target_path, predicted_path):
    """Computes and prints the overall classification error and precision, recall, F-score over punctuations."""
    mappings, counter, t_i, p_i = {}, 0, 0, 0
    total_correct, correct, substitutions, deletions, insertions = 0, 0.0, 0.0, 0.0, 0.0
    true_pos, false_pos, false_neg = {}, {}, {}
    with codecs.open(target_path, "r", "utf-8") as f_target, codecs.open(predicted_path, "r", "utf-8") as f_predict:
        target_stream = f_target.read().split()
        predict_stream = f_predict.read().split()
        while True:
            if t_i<len(target_stream) and PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i]) in PUNCTUATION_VOCABULARY:
                # skip multiple consecutive punctuations
                target_punct = " "
                while t_i<len(target_stream) and PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i]) in PUNCTUATION_VOCABULARY:
                    target_punct = PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i])
                    target_punct = mappings.get(target_punct, target_punct)
                    t_i += 1
            else:
                target_punct = " "
            if  p_i<len(predict_stream) and predict_stream[p_i] in PUNCTUATION_VOCABULARY:
                predicted_punct = mappings.get(predict_stream[p_i], predict_stream[p_i])
                p_i += 1
            else:
                predicted_punct = " "
            is_correct = target_punct == predicted_punct
            counter += 1
            total_correct += is_correct
            if predicted_punct == " " and target_punct != " ":
                deletions += 1
            elif predicted_punct != " " and target_punct == " ":
                insertions += 1
            elif predicted_punct != " " and target_punct != " " and predicted_punct == target_punct:
                correct += 1
            elif predicted_punct != " " and target_punct != " " and predicted_punct != target_punct:
                substitutions += 1
            true_pos[target_punct] = true_pos.get(target_punct, 0.0) + float(is_correct)
            false_pos[predicted_punct] = false_pos.get(predicted_punct, 0.) + float(not is_correct)
            false_neg[target_punct] = false_neg.get(target_punct, 0.) + float(not is_correct)
            # assert target_stream[t_i] == predict_stream[p_i] or predict_stream[p_i] == "<unk>", \
            #     "File: %s \nError: %s (%s) != %s (%s) \nTarget context: %s \nPredicted context: %s" % \
            #     (target_path, target_stream[t_i], t_i, predict_stream[p_i], p_i,
            #      " ".join(target_stream[t_i - 2:t_i + 2]), " ".join(predict_stream[p_i - 2:p_i + 2]))
            t_i += 1
            p_i += 1
            if t_i >= len(target_stream) - 1 and p_i >= len(predict_stream) - 1:
                break
    overall_tp, overall_fp, overall_fn = 0.0, 0.0, 0.0
    out_str = "-" * 46 + "\n"
    out_str += "{:<16} {:<9} {:<9} {:<9}\n".format("PUNCTUATION", "PRECISION", "RECALL", "F-SCORE")
    for p in PUNCTUATION_VOCABULARY:
        if p == SPACE:
            continue
        overall_tp += true_pos.get(p, 0.0)
        overall_fp += false_pos.get(p, 0.0)
        overall_fn += false_neg.get(p, 0.0)
        punctuation = p
        precision = (true_pos.get(p, 0.0) / (true_pos.get(p, 0.0) + false_pos[p])) if p in false_pos else nan
        recall = (true_pos.get(p, 0.0) / (true_pos.get(p, 0.0) + false_neg[p])) if p in false_neg else nan
        f_score = (2. * precision * recall / (precision + recall)) if (precision + recall) > 0 else nan
        out_str += u"{:<16} {:<9} {:<9} {:<9}\n".format(punctuation, "{:.2f}".format(precision * 100),
                                                        "{:.2f}".format(recall * 100),
                                                        "{:.2f}".format(f_score * 100))
    out_str += "-" * 46 + "\n"
    pre = overall_tp / (overall_tp + overall_fp) if overall_fp else nan
    rec = overall_tp / (overall_tp + overall_fn) if overall_fn else nan
    f1 = (2. * pre * rec) / (pre + rec) if (pre + rec) else nan
    out_str += "{:<16} {:<9} {:<9} {:<9}\n".format("Overall", "{:.2f}".format(pre * 100),
                                                   "{:.2f}".format(rec * 100), "{:.2f}".format(f1 * 100))
    err = round((100.0 - float(total_correct) / float(counter - 1) * 100.0), 2)
    ser = round((substitutions + deletions + insertions) / (correct + substitutions + deletions) * 100, 1)
    out_str += "ERR: %s%%\n" % err
    out_str += "SER: %s%%" % ser
    return out_str, f1, err, ser
if __name__ == "__main__":

    # out_str, f1, err, ser = compute_score(
    #     '/data/dh/neural_sequence_labeling-master/data/raw/LREC_converted/ref.txt',
    #     '/data/dh/neural_sequence_labeling-master/bert/output/label_test.txt')
    #
    # score = {"F1": f1, "ERR": err, "SER": ser}
    # print(out_str)
    # print(score)
    # tf.logging.info("\nEvaluate on {}:\n{}\n".format('ref', out_str))
    main()