import os
import collections
import tensorflow as tf
from .bert import tokenization

from gooey import Gooey, GooeyParser

from .test_config import test_config_parser, test_config
from .bert.run_squad import FeatureWriter, convert_examples_to_features, input_fn_builder, write_predictions


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def test_parser(
        parser: GooeyParser = GooeyParser(),
        title="train Setting",
        description="") -> GooeyParser:

    subs = parser.add_subparsers()

    test_parser = subs.add_parser('test')
    test_config_parser(test_parser)

    return parser


def test(model,
         model_config,
         dataset_test,
         stream=None):
    eval_examples = dataset_test

    tokenizer = tokenization.FullTokenizer(
        vocab_file=model_config.VOCAB_FILE, do_lower_case=model_config.DO_LOWER_CASE)

    eval_writer = FeatureWriter(
        filename=os.path.join(model_config.OUTPUT_DIR, "eval.tf_record"),
        is_training=False)
    eval_features = []

    def append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)

    convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=model_config.MAX_SEQ_LENGTH,
        doc_stride=model_config.DOC_STRIDE,
        max_query_length=model_config.MAX_QUERY_LENGTH,
        is_training=False,
        output_fn=append_feature)
    eval_writer.close()

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", model_config.PREDICT_BATCH_SIZE)

    predict_input_fn = input_fn_builder(
        input_file=eval_writer.filename,
        seq_length=model_config.MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)

    # If running eval on the TPU, you will need to specify the number of steps.
    all_results = []
    for result in model.predict(
            predict_input_fn, yield_single_examples=True):
        if len(all_results) % 1000 == 0:
            tf.logging.info("Processing example: %d" % (len(all_results)))
        unique_id = int(result["unique_ids"])
        start_logits = [float(x) for x in result["start_logits"].flat]
        end_logits = [float(x) for x in result["end_logits"].flat]
        all_results.append(
            RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits))

    output_prediction_file = os.path.join(model_config.OUTPUT_DIR, "predictions.json")
    output_nbest_file = os.path.join(model_config.OUTPUT_DIR, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(model_config.OUTPUT_DIR, "null_odds.json")

    write_predictions(eval_examples, eval_features, all_results,
                      model_config.N_BEST_SIZE, model_config.MAX_ANSWER_LENGTH,
                      model_config.DO_LOWER_CASE, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file)
