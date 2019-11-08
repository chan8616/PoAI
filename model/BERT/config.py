class Config(object):
    NAME = None

    '''
    REQUIRES
    '''
    # The config json file corresponding to the pre-trained BERT model. REQUIRED
    # This specifies the model architecture.
    BERT_CONFIG_FILE = None

    # The vocabulary file that the BERT model was trained on. REQUIRED
    VOCAB_FILE = None

    # The output directory where the model checkpoints will be written. REQUIRED
    OUTPUT_DIR = None

    '''
    TRAIN_SETTING
    '''
    # SQuAD json for training. E.g., train-v1.1.json
    TRAIN_FILE = None

    # SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json
    PREDICT_FILE = None

    # Initial checkpoint (usually from a pre-trained BERT model).
    INIT_CHECKPOINT = None

    # Whether to lower case the input text.
    # Should be True for uncased models and False for cased models.
    DO_LOWER_CASE = True

    # The maximum total input sequence length after WordPiece tokenization.
    # Sequences longer than this will be truncated, and sequences shorter than this will be padded.
    MAX_SEQ_LENGTH = 384

    # When splitting up a long document into chunks, how much stride to take between chunks.
    DOC_STRIDE = 128

    # The maximum number of tokens for the question. Questions longer than this will be truncated to this length.
    MAX_QUERY_LENGTH = 64

    # Whether to run training.
    DO_TRAIN = False

    # Whether to run eval on the dev set.
    DO_PREDICT = False

    # Total batch size for training.
    TRAIN_BATCH_SIZE = 32

    # Total batch size for predictions.
    PREDICT_BATCH_SIZE = 8

    # Total number of training epochs to perform.
    NUM_TRAIN_EPOCHS = 3

    # The initial learning rate for Adam.
    LEARNING_RATE = 5e-5

    # Proportion of training to perform linear learning rate warmup for.
    # E.g., 0.1 = 10% of training.
    WARMUP_PROPORTION = 0.1

    # How often to save the model checkpoint.
    SAVE_CHECKPOINTS_STEPS = 1000

    # How many steps to make in each estimator call.
    ITERATION_PER_LOOP = 1000

    # "The total number of n-best predictions to generate in the nbest_predictions.json output file."
    N_BEST_SIZE = 20

    # The maximum length of an answer that can be generated.
    # This is needed because the start and end predictions are not conditioned on one another.
    MAX_ANSWER_LENGTH = 30

    # If true, all of the warnings related to data processing will be printed.
    # A number of warnings are expected for a normal SQuAD evaluation.
    VERBOSE_LOGGING = False

    # If true, the SQuAD examples contain some that do not have an answer.
    VERSION_2_WITH_NEGATIVE = False

    # If null_score - best_non_null is greater than the threshold predict null.
    NULL_SCORE_DIFF_THRESHOLD = 0.0

    '''
    TPU SETTING
    '''
    # Whether to use TPU or GPU/CPU."
    USE_TPU = False

    # The Cloud TPU to use for training.
    # This should be either the name used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.
    TPU_NAME = None

    # Only used if `use_tpu` is True. Total number of TPU cores to use.
    NUM_TPU_CORES = 8

    # [Optional]
    # GCE zone where the Cloud TPU is located in.
    # If not specified, we will attempt to automatically detect the GCE project from metadata.
    TPU_ZONE = None

    # [Optional]
    # Project name for the Cloud TPU-enabled project.
    # If not specified, we will attempt to automatically detect the GCE project from metadata."
    GCP_PROJECT = None

    # [Optional]
    # TensorFlow master URL.
    MASTER = None

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
