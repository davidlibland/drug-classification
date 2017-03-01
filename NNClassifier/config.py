import os

base_dir = os.path.abspath(os.path.dirname(__file__))
sampling_model_dir = 'Saved_Model_Dir'
sampling_model = 'model.ckpt'
args_file = 'args.pkl'
SAMPLE_LENGTH = 32

NUM_DRUG_CLASSES = 33

# TRAINING
MAX_EPOCHS = 30
NUM_STEPS = 32 #16
NUM_LAYERS = 2
BATCH_SIZE = 32
KEEP_PROB = 0.5
HIDDEN_SIZE = 512
WORD_EMBEDDING_SIZE=64
INIT_SCALE = 0.05
LEARNING_RATE = .01

# DEVELOPMENT TESTING:
MAX_EPOCHS = 50
NUM_STEPS = 32 # 8
NUM_LAYERS = 1
BATCH_SIZE = 4
KEEP_PROB = 0.5
HIDDEN_SIZE = 32
WORD_EMBEDDING_SIZE=16