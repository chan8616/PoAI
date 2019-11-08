import os
import tensorflow as tf
import random
from .bert import tokenization
from gooey import GooeyParser

from .train_config import train_config_parser
from ..utils.stream_callbacks import tf_logger

from .bert.run_squad import FeatureWriter, convert_examples_to_features, input_fn_builder


def train_parser(
        parser: GooeyParser = GooeyParser(),
        title="train Setting",
        description="") -> GooeyParser:

    subs = parser.add_subparsers()

    train_parser_ = subs.add_parser('train')
    train_config_parser(train_parser_)

    return parser


def train(t_config):
    tf.logging.set_verbosity(tf.logging.INFO)
    model, model_config, dataset_train, dataset_val, stream = t_config

    num_train_steps = int(len(dataset_train) / model_config.TRAIN_BATCH_SIZE * model_config.NUM_TRAIN_EPOCHS)

    # Pre-shuffle the input to avoid having to make a very large shuffle buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(dataset_train)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=model_config.VOCAB_FILE, do_lower_case=model_config.DO_LOWER_CASE)

    # We write to a temporary file to avoid storing very large constant tensors in memory.
    train_writer = FeatureWriter(
        filename=os.path.join(model_config.OUTPUT_DIR, "train.tf_record"),
        is_training=True)
    convert_examples_to_features(
        examples=dataset_train,
        tokenizer=tokenizer,
        max_seq_length=model_config.MAX_SEQ_LENGTH,
        doc_stride=model_config.DOC_STRIDE,
        max_query_length=model_config.MAX_QUERY_LENGTH,
        is_training=True,
        output_fn=train_writer.process_feature)
    train_writer.close()

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", len(dataset_train))
    tf.logging.info("  Num split examples = %d", train_writer.num_features)
    tf.logging.info("  Batch size = %d", model_config.TRAIN_BATCH_SIZE)
    tf.logging.info("  Num steps = %d", num_train_steps)
    del dataset_train

    train_input_fn = input_fn_builder(
        input_file=train_writer.filename,
        seq_length=model_config.MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=True)

    # accuracy = tf.metrics.accuracy()
    # train_hook = tf_logger(stream)

    model.train(input_fn=train_input_fn, max_steps=num_train_steps)

