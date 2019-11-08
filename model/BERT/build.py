from argparse import Namespace
from pathlib import Path
import random
from gooey import Gooey, GooeyParser
import tensorflow as tf
from .bert import modeling, tokenization

from .bert.run_squad import model_fn_builder
from .config import Config

from .build_config import build_config_parser


def build_parser(
        parser: GooeyParser = GooeyParser(),
        title="Build Setting",
        description="") -> GooeyParser:

    subs = parser.add_subparsers()

    build_parser_ = subs.add_parser('build')
    build_config_parser(build_parser_)

    return parser


#  def build(build_cmd, build_args):
#      return config(build_args)

#      build_config, args = build_config(config_args)
#      return modellib.MaskRCNN(mode=run_cmd, config=build_config,
#                               model_dir=run_args.model_dir)
#      return build_config(args)


def build(model_config: Config, dataset_size=None):

    bert_config = modeling.BertConfig.from_json_file(model_config.BERT_CONFIG_FILE)

    validate_config_or_throw(model_config, bert_config)

    tf.gfile.MakeDirs(model_config.OUTPUT_DIR)

    tpu_cluster_resolver = None
    if model_config.USE_TPU and model_config.TPU_NAME:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            model_config.TPU_NAME, zone=model_config.TPU_ZONE, project=model_config.GCP_PROJECT)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config_ = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=model_config.MASTER,
        model_dir=model_config.OUTPUT_DIR,
        save_checkpoints_steps=model_config.SAVE_CHECKPOINTS_STEPS,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=model_config.ITERATION_PER_LOOP,
            num_shards=model_config.NUM_TPU_CORES,
            per_host_input_for_training=is_per_host))

    num_train_steps = None
    num_warmup_steps = None
    if model_config.DO_TRAIN:
        num_train_steps = int(
            dataset_size / model_config.TRAIN_BATCH_SIZE * model_config.NUM_TRAIN_EPOCHS)
        num_warmup_steps = int(num_train_steps * model_config.WARMUP_PROPORTION)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=model_config.INIT_CHECKPOINT,
        learning_rate=model_config.LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=model_config.USE_TPU,
        use_one_hot_embeddings=model_config.USE_TPU)

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=model_config.USE_TPU,
        model_fn=model_fn,
        config=run_config_,
        train_batch_size=model_config.TRAIN_BATCH_SIZE,
        predict_batch_size=model_config.PREDICT_BATCH_SIZE)

    return estimator


def validate_config_or_throw(run_config: Config, bert_config):
    tokenization.validate_case_matches_checkpoint(run_config.DO_LOWER_CASE,
                                                  run_config.INIT_CHECKPOINT)

    if not run_config.DO_TRAIN and not run_config.DO_PREDICT:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if run_config.DO_TRAIN:
        if not run_config.TRAIN_FILE:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if run_config.DO_PREDICT:
        if not run_config.PREDICT_FILE:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if run_config.MAX_SEQ_LENGTH > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (run_config.MAX_SEQ_LENGTH, bert_config.max_position_embeddings))

    if run_config.MAX_SEQ_LENGTH <= run_config.MAX_QUERY_LENGTH + 3:
        raise ValueError(
            "The max_seq_length (%d) must be greater than max_query_length "
            "(%d) + 3" % (run_config.MAX_SEQ_LENGTH, run_config.MAX_QUERY_LENGTH))
