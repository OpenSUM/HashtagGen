PRETRAIN_WORD_EMBEDDING = 1
TRAINABLE_WORD_EMBEDDING = 2
BOTH_WORD_EMBEDDING = 3

LTP_DIR = r'./ltp_data_v3.4.0'
# BPE_CODEC_FILE = r'/home/LAB/tangb/projects/bishe/data/news/bpe/train.codec'
# BPE_VOCAB_FILE = r'/home/LAB/tangb/projects/bishe/data/news/bpe/vocab'
# BPE_VOCAB_THRESHOLD = 30

CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
SENTEN_TOKEN = '[SENTEN]'

EXP_DIR = './experiments'

EPSILON = 1e-10

CHECKPOINTS_MAX_TO_KEEP = 999

HALVE_BERT_LR = False
HALVE_OTHER_LR = False
# 每隔多少batch测试一次rouge2
CHECK_GLOBAL_STEPS = 5
#每隔多少batch 输出一下
PRINT_STEPS = 1
#每隔多少batch学习率减半（通过上边的减半的参数控制，减半哪一部分学习率）
HALVE_LR_STEPS = CHECK_GLOBAL_STEPS * 5
MIN_LEARNING_RATE = 1e-10

SAMPLE_LIMIT = None  # DEPRECATED.
#训练集最大的数据量，用全部数据的话，可以=None
TRAIN_SAMPLE_LIMIT = None
EVAL_SAMPLE_LIMIT = None
MAX_SAMPLE_NUM = 2400591
