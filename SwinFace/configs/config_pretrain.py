from easydict import EasyDict as edict
from torchvision.transforms import InterpolationMode

config = edict()
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
# Landmark datasets
config.CelebA_data = "/experiments/shared_face_rec_pep/CelebA/"
config.CelebA_train_label = "/experiments/shared_face_rec_pep/CelebA/list_landmarks_train.txt"
config.CelebA_test_label = "/experiments/shared_face_rec_pep/CelebA/list_landmarks_test.txt"
config.CelebA_val_label = "/experiments/shared_face_rec_pep/CelebA/list_landmarks_val.txt"

config.MTFL_train_data = "/experiments/shared_face_rec_pep/MTFL"
config.MTFL_train_label = "/experiments/shared_face_rec_pep/MTFL/training.txt"
config.MTFL_test_data = "/experiments/shared_face_rec_pep/MTFL"
config.MTFL_test_label = "/experiments/shared_face_rec_pep/MTFL/testing.txt"

# Head pose datasets
config.AFLW_train_data = "/experiments/shared_face_rec_pep/AFLW/data/flickr"
config.AFLW_train_label = "/experiments/shared_face_rec_pep/AFLW/data/flickr/aflw_train.txt"
config.AFLW_val_data = "/experiments/shared_face_rec_pep/AFLW/data/flickr/"
config.AFLW_val_label = "/experiments/shared_face_rec_pep/AFLW/data/flickr/aflw_val.txt"
config.AFLW_test_data = "/experiments/shared_face_rec_pep/AFLW/data/flickr"
config.AFLW_test_label = "/experiments/shared_face_rec_pep/AFLW/data/flickr/aflw_test.txt"

config.WLP_data_train = "/experiments/shared_face_rec_pep/300W_LP21/train"
config.WLP_data_test = "/experiments/shared_face_rec_pep/300W_LP21/test"

# Expression datasets
config.RAF_data_train = "/experiments/shared_face_rec_pep/RAF_DB/train"
config.RAF_label_train = "/experiments/shared_face_rec_pep/RAF_DB/train_labels.csv"
config.RAF_data_test = "/experiments/shared_face_rec_pep/RAF_DB/test"
config.RAF_label_test = "/experiments/shared_face_rec_pep/RAF_DB/test_labels.csv"

config.AffectNet_train_data = "/experiments/shared_face_rec_pep/AffectNet+/human_annotated/train_set/images"
config.AffectNet_train_label = "/experiments/shared_face_rec_pep/AffectNet+/human_annotated/train_set/annotations"
config.AffectNet_val_data = "/experiments/shared_face_rec_pep/AffectNet+/human_annotated/validation_set/images"
config.AffectNet_val_label = "/experiments/shared_face_rec_pep/AffectNet+/human_annotated/validation_set/annotations"

config.INTERPOLATION = InterpolationMode.BICUBIC
config.RAF_NUM_CLASSES = 7
config.RAF_LABEL_SMOOTHING = 0
config.AUG_COLOR_JITTER = 0
config.AUG_AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
config.AUG_REPROB = 0
config.AUG_REMODE = 'pixel'
config.AUG_RECOUNT = 0
config.AUG_MIXUP = 0.0
config.AUG_CUTMIX = 0.0
config.AUG_CUTMIX_MINMAX = None
config.AUG_MIXUP_PROB = 0.0
config.AUG_MIXUP_SWITCH_PROB = 0.0
config.AUG_MIXUP_MODE = 'batch'
config.AUG_SCALE_SET = False
config.AUG_SCALE_SCALE = (1.0, 1.0)
config.AUG_SCALE_RATIO = (1.0, 1.0)

# Data loading settings
config.img_size = 112
config.batch_size = 64
config.val_batch_size = 64
config.landmark_bz = 64
config.expression_bz = 64
config.head_pose_bz = 64
config.num_workers = 1
config.train_num_workers = 8
config.train_pin_memory = True
config.val_num_workers = 1
config.val_pin_memory = True

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------

config.network = "swin_t"
config.embedding_size = 512

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------

# Resume
config.resume = False

# Epoch num
config.warmup_epoch = 5
config.num_epoch = 40

# For SGD
#config.optimizer = "sgd"
#config.lr = 0.1
#config.momentum = 0.9
#config.weight_decay = 5e-4

# For AdamW
config.optimizer = "adamw"
config.lr = 5e-4
config.weight_decay = 0.05

# Learning rate
config.lr_name = 'cosine'
config.warmup_lr = 5e-7
config.min_lr = 5e-6
config.decay_epoch = 10 # Epoch interval to decay LR, used in StepLRScheduler
config.decay_rate = 0.1 # LR decay rate, used in StepLRScheduler

# loss
config.margin_list = (1.0, 0.0, 0.4)
config.sample_rate = 0.3
config.interclass_filtering_threshold = 0
config.analysis_loss_weights = [1.0, 4.0, 10.0]

# Others
config.fp16 = True
config.dali = False # For Large Scale Dataset, such as WebFace42M
config.seed = 2048

# -----------------------------------------------------------------------------
# Output and Saving
# -----------------------------------------------------------------------------

config.save_all_states = True
config.output = "/home/lslopes/Swin-Transformer-Object-Detection/SwinFace/output" ####Path for Output

config.verbose = 2000
config.frequent = 10

config.resume = True
config.resume_epoch = 24
config.init = False
config.init_model = "/home/lslopes/Swin-Transformer-Object-Detection/SwinFace/output/checkpoint_step_99999_gpu_0_EMA.pt"

