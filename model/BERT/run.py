import datetime
from pathlib import Path
from gooey import GooeyParser
from .config import Config

from . import (build, build_config,
               train, train_config,
               test, test_config,
               generator, generator_config,
               )


def run_parser(
        parser: GooeyParser = GooeyParser(),
        title="Train Setting",
        description="") -> GooeyParser:

    subs = parser.add_subparsers()

    test_parser = subs.add_parser('test')
    test_config.test_config_parser(test_parser)

    train_parser = subs.add_parser('train')
    train_config.train_config_parser(train_parser)

    return parser


# Should be fixed. It is directly used in gui/frame.py
def run(config):
    print(config)
    build_cmds, build_args, run_cmds, run_args, generator_cmds, generator_args, stream = config
    build_cmd = build_cmds[0]
    run_cmd = run_cmds[0]
    generator_cmd = generator_cmds[0]
    stream.put(('Generating Dataset...', None, None))
    #  build_config = build_config.build_config(build_args)
    #  generator_config = generator_config.generator_config(generator_args)
    dataset, dataset_val = generator.generator(generator_cmd, generator_args)

    model_config = None
    stream.put(('Generating...', None, None))
    if 'train' in run_cmd:
        model_config = model_config_builder('train',
                                            build_config.build_config(build_args),
                                            train_config.train_config(run_args),
                                            generator_config.generator_config(generator_args))
        #  train_config = train_config.train_config(run_args)
        model = build.build(model_config, len(dataset))
    elif 'test' == run_cmd:
        model_config = model_config_builder('test',
                                            build_config.build_config(build_args),
                                            train_config.train_config(run_args),
                                            generator_config.generator_config(generator_args))
        model = build.build(model_config)
    else:
        raise AttributeError("run_cmd must be train or test!")

    #  config = model_config(build_config, train_config),
    #  model = modellib.MaskRCNN(model=run_cmd,
    #                            config=config,
    #                            model_dir=MODEL_DIR.joinpath(args.save_dir))
    #  setting = train.train_setting(model, run_args)
    if 'train' in run_cmd:
        print('before train')
        stream.put(('Training...', None, None))
        return train.train((model, model_config, dataset, dataset_val, stream))
    elif 'test' == run_cmd:
        print('before test')
        stream.put(('Testing...', None, None))
        return test.test(model, model_config, dataset)
    stream.put(('End', None, None))


def model_config_builder(mode, build_config_: Config, run_config: Config, gen_config: Config):
    class ModelConfig(Config):
        NAME = build_config_.NAME
        BERT_CONFIG_FILE = build_config_.BERT_CONFIG_FILE
        VOCAB_FILE = build_config_.VOCAB_FILE
        OUTPUT_DIR = build_config_.OUTPUT_DIR

        INIT_CHECKPOINT = run_config.INIT_CHECKPOINT
        if mode == 'train':
            DO_TRAIN = True
            DO_PREDICT = False
        elif mode == 'test':
            DO_TRAIN = False
            DO_PREDICT = True
        TRAIN_BATCH_SIZE = run_config.TRAIN_BATCH_SIZE
        PREDICT_BATCH_SIZE = run_config.PREDICT_BATCH_SIZE
        NUM_TRAIN_EPOCHS = run_config.NUM_TRAIN_EPOCHS
        LEARNING_RATE = run_config.LEARNING_RATE
        WARMUP_PROPORTION = run_config.WARMUP_PROPORTION
        SAVE_CHECKPOINTS_STEPS = run_config.SAVE_CHECKPOINTS_STEPS
        ITERATION_PER_LOOP = run_config.ITERATION_PER_LOOP
        VERBOSE_LOGGING = run_config.VERBOSE_LOGGING
        NULL_SCORE_DIFF_THRESHOLD = run_config.NULL_SCORE_DIFF_THRESHOLD

        TRAIN_FILE = gen_config.TRAIN_FILE
        PREDICT_FILE = gen_config.PREDICT_FILE
        DO_LOWER_CASE = gen_config.DO_LOWER_CASE
        MAX_SEQ_LENGTH = gen_config.MAX_SEQ_LENGTH
        DOC_STRIDE = gen_config.DOC_STRIDE
        MAX_QUERY_LENGTH = gen_config.MAX_QUERY_LENGTH
        MAX_ANSWER_LENGTH = gen_config.MAX_ANSWER_LENGTH
        VERSION_2_WITH_NEGATIVE = gen_config.VERSION_2_WITH_NEGATIVE

    return ModelConfig()

