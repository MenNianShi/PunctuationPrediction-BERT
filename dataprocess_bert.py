import tensorflow as tf
import os
from data.bert_prepro import process_data
from utils import batchnize_dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# dataset parameters
tf.flags.DEFINE_string("language", "english", "language")  # used for inference, indicated the source language
tf.flags.DEFINE_string("raw_path", "data/raw/LREC_converted", "path to raw dataset")
tf.flags.DEFINE_string("save_path", "data/dataset/lrec", "path to save dataset")
tf.flags.DEFINE_string("glove_name", "840B", "glove embedding name")
# glove embedding path
glove_path = '/data/dh/glove/glove.840B.300d.txt'
#glove_path = os.path.join(os.path.expanduser(''), "utilities", "embeddings", "glove.{}.{}d.txt")
tf.flags.DEFINE_string("glove_path", glove_path, "glove embedding path")
tf.flags.DEFINE_integer("max_vocab_size", 50000, "maximal vocabulary size")
tf.flags.DEFINE_integer("max_sequence_len", 200, "maximal sequence length allowed")
tf.flags.DEFINE_integer("min_word_count", 1, "minimal word count in word vocabulary")
tf.flags.DEFINE_integer("min_char_count", 10, "minimal character count in char vocabulary")

# dataset for train, validate and test
tf.flags.DEFINE_string("vocab", "data/dataset/lrec/bert_tvocab.json", "path to the word and tag vocabularies")
tf.flags.DEFINE_string("train_set", "data/dataset/lrec/bert_train.json", "path to the training datasets")
tf.flags.DEFINE_string("dev_set", "data/dataset/lrec/bert_dev.json", "path to the development datasets")
tf.flags.DEFINE_string("dev_text", "data/raw/LREC_converted/dev.txt", "path to the development text")
tf.flags.DEFINE_string("ref_set", "data/dataset/lrec/bert_ref.json", "path to the ref test datasets")
tf.flags.DEFINE_string("ref_text", "data/raw/LREC_converted/ref.txt", "path to the ref text")
tf.flags.DEFINE_string("asr_set", "data/dataset/lrec/bert_asr.json", "path to the asr test datasets")
tf.flags.DEFINE_string("asr_text", "data/raw/LREC_converted/asr.txt", "path to the asr text")
tf.flags.DEFINE_string("pretrained_emb", "data/dataset/lrec/glove_emb.npz", "pretrained embeddings")
tf.flags.DEFINE_string("pos_emb", "data/dataset/lrec/pos_emb.npz", "pretrained embeddings")


tf.flags.DEFINE_string("cell_type", "lstm", "RNN cell for encoder and decoder: [lstm | gru], default: lstm")
tf.flags.DEFINE_integer("num_layers", 4, "number of rnn layers")
tf.flags.DEFINE_boolean("use_pretrained", True, "use pretrained word embedding")
tf.flags.DEFINE_boolean("tuning_emb", False, "tune pretrained word embedding while training")
tf.flags.DEFINE_integer("emb_dim", 300, "embedding dimension for encoder and decoder input words/tokens")
tf.flags.DEFINE_boolean("use_chars", True, "use char embeddings")
tf.flags.DEFINE_boolean("use_pos", False, "use char embeddings")
tf.flags.DEFINE_integer("char_emb_dim", 50, "character embedding dimension")
tf.flags.DEFINE_boolean("use_highway", True, "use highway network")
tf.flags.DEFINE_integer("highway_layers", 2, "number of layers for highway network")
tf.flags.DEFINE_boolean("use_crf", True, "use CRF decoder")
tf.flags.DEFINE_string("char_represent_method", "cnn", "method to represent char embeddings: [rnn | cnn]")
tf.flags.DEFINE_integer("char_num_units", 50, "character rnn hidden units")


# training parameters
tf.flags.DEFINE_float("lr", 0.001, "learning rate")
tf.flags.DEFINE_string("optimizer", "adam", "optimizer: [adagrad | sgd | rmsprop | adadelta | adam], default: adam")
tf.flags.DEFINE_boolean("use_lr_decay", True, "apply learning rate decay for each epoch")
tf.flags.DEFINE_float("lr_decay", 0.5, "learning rate decay factor")
tf.flags.DEFINE_float("l2_reg", None, "L2 norm regularization")
tf.flags.DEFINE_float("minimal_lr", 1e-5, "minimal learning rate")
tf.flags.DEFINE_float("grad_clip", 2.0, "maximal gradient norm")
tf.flags.DEFINE_float("keep_prob", 0.75, "dropout keep probability for embedding while training")
tf.flags.DEFINE_integer("batch_size", 32, "batch size")
tf.flags.DEFINE_integer("epochs", 15, "train epochs")
tf.flags.DEFINE_integer("max_to_keep", 3, "maximum trained models to be saved")
tf.flags.DEFINE_integer("no_imprv_tolerance", 10, "no improvement tolerance")
tf.flags.DEFINE_string("checkpoint_path", "ckpt/punctuator/", "path to save models checkpoints")
tf.flags.DEFINE_string("summary_path", "ckpt/punctuator/summary/", "path to save summaries")
tf.flags.DEFINE_string("model_name", "attentive_punctuator_model", "models name")



# convert parameters to dict
config = tf.flags.FLAGS.flag_values_dict()


print("")
#create dataset from raw data files

process_data(config)

