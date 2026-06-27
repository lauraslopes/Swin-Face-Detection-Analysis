from easydict import EasyDict as edict

config = edict()
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------

# Recognition val targets
config.val_targets = ['lfw', 'cfp_fp', "agedb_30", 'calfw', 'cplfw']

# Recognition data
config.rec = "<your path>/data/faces_emore/"  #### Path for the training dataset
config.num_classes = 85742
config.num_image = 5822653

# Analysis data
config.age_gender_data_path = "<your path>/data/AIO_train"
config.age_gender_data_list = ["IMDB", "WIKI", "Adience", "MORPH"]

config.CelebA_train_data = "<your path>/data/CelebA/data"
config.CelebA_train_label = "<your path>/data/AIO_train/CelebA/label.txt"
config.CelebA_val_data = "<your path>/data/CelebA/data"
config.CelebA_val_label = "<your path>/data/AIO_val/CelebA/label.txt"
config.CelebA_test_data = "<your path>/data/CelebA/data"
config.CelebA_test_label = "<your path>/data/AIO_test/CelebA/label.txt"

config.head_pose_train_data = "/experiments/shared_face_rec_pep/AFLW/data/flickr"
config.head_pose_train_label = "/experiments/shared_face_rec_pep/AFLW/data/flickr/aflw_train.txt"
config.head_pose_val_data = "/experiments/shared_face_rec_pep/AFLW/data/flickr/"
config.head_pose_val_label = "/experiments/shared_face_rec_pep/AFLW/data/flickr/aflw_val.txt"
config.head_pose_test_data = "/experiments/shared_face_rec_pep/AFLW/data/flickr"
config.head_pose_test_label = "/experiments/shared_face_rec_pep/AFLW/data/flickr/aflw_test.txt"

config.FGnet_data = "<your path>/data/AIO_val/FGnet/data"
config.FGnet_label = "<your path>/data/AIO_val/FGnet/label.txt"

config.RAF_data_train = "/experiments/shared_face_rec_pep/RAF_DB/train"
config.RAF_data_test = "/experiments/shared_face_rec_pep/RAF_DB/test"
config.RAF_label_train = "/experiments/shared_face_rec_pep/RAF_DB/train_labels.csv"
config.RAF_label_test = "/experiments/shared_face_rec_pep/RAF_DB/test_labels.csv"

# AffectNet human_annotated train/validation layout
# AffectNet human_annotated layout: separate images and per-image json annotations
config.AffectNet_train_data = "/experiments/shared_face_rec_pep/AffectNet+/human_annotated/train_set/images"
config.AffectNet_train_label = "/experiments/shared_face_rec_pep/AffectNet+/human_annotated/train_set/annotations"
config.AffectNet_val_data = "/experiments/shared_face_rec_pep/AffectNet+/human_annotated/validation_set/images"
config.AffectNet_val_label = "/experiments/shared_face_rec_pep/AffectNet+/human_annotated/validation_set/annotations"

config.W_LP_data = "/experiments/shared_face_rec_pep/300W_LP21"

config.LAP_train_data = "<your path>/data/AIO_test/LAP_finetuning/data"
config.LAP_train_label = "<your path>/data/AIO_test/LAP_finetuning/label.csv"
config.LAP_test_data = "<your path>/data/AIO_test/LAP_test/data"
config.LAP_test_label = "<your path>/data/AIO_test/LAP_test/label.csv"

config.CLAP_train_data = "<your path>/data/AIO_test/LAP_finetuning/data"
config.CLAP_train_label = "<your path>/data/AIO_test/LAP_finetuning/label.csv"
config.CLAP_val_data = "<your path>/data/AIO_test/LAP_test/data"
config.CLAP_val_label = "<your path>/data/AIO_test/LAP_test/label.csv"
config.CLAP_test_data = "<your path>/data/LAP_test/test"
config.CLAP_test_label = "<your path>/data/LAP_test/test.csv"

# Data loading settings
config.img_size = 112
config.batch_size = 128
config.recognition_bz = 32
config.age_gender_bz = 32
config.CelebA_bz = 32
config.expression_bz = 32
config.head_pose_bz = 32
config.train_num_workers = 1
config.train_pin_memory = True

config.val_batch_size = 64
config.val_num_workers = 1
config.val_pin_memory = True

# Data argument

config.INTERPOLATION = 'bicubic'
config.RAF_NUM_CLASSES = 7
# Label Smoothing
config.RAF_LABEL_SMOOTHING = 0.1

config.AUG_COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
config.AUG_AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
config.AUG_REPROB = 0.25
# Random erase mode
config.AUG_REMODE = 'pixel'
# Random erase count
config.AUG_RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
config.AUG_MIXUP = 0.0 #0.8
# Cutmix alpha, cutmix enabled if > 0
config.AUG_CUTMIX = 0.0 #1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
config.AUG_CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
config.AUG_MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
config.AUG_MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
config.AUG_MIXUP_MODE = 'batch'

config.AUG_SCALE_SET = True
config.AUG_SCALE_SCALE = (1.0, 1.0)
config.AUG_SCALE_RATIO = (1.0, 1.0)

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------

config.network = "swin_t"

config.fam_kernel_size=3
config.fam_in_chans=2112
config.fam_conv_shared=False
config.fam_conv_mode="split"
config.fam_channel_attention="CBAM"
config.fam_spatial_attention=None
config.fam_pooling="max"
config.fam_la_num_list=[2 for j in range(3)]
config.fam_feature="all"
config.fam = "3x3_2112_F_s_C_N_max"

config.embedding_size = 512
config.analysis_branch_num = 3

# Training weights for the 3 active analysis tasks: Expression, Landmarks, Pose
config.analysis_loss_weights = [1.0, 1.0, 1.0]

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------

# Resume and init
config.resume = False
config.resume_step = 3999
config.init = False
config.init_model = "/home/lslopes/Swin-Transformer-Object-Detection/SwinFace/output/checkpoint_step_79999_gpu_0.pt"

# Step num
config.warmup_step = 8000
config.total_step = 80000

# SGD optimizer
#config.optimizer = "sgd"
#config.lr = 0.1
#config.momentum = 0.9
#config.weight_decay = 5e-4

# AdamW optimizer
config.optimizer = "adamw"
config.lr = 5e-4
config.weight_decay = 0.05

# Learning rate
config.lr_name = 'cosine'
config.warmup_lr = 5e-7
config.min_lr = 5e-6
config.decay_epoch = 10 # Epoch interval to decay LR, used in StepLRScheduler
config.decay_rate = 0.1 # LR decay rate, used in StepLRScheduler

# Recognition loss
config.margin_list = (1.0, 0.0, 0.4)
config.sample_rate = 0.3 # Partial FC
config.interclass_filtering_threshold = 0 # Partial FC

# Loss weight
config.recognition_loss_weight = 1.0
config.analysis_loss_weights = [1.0, 1.0, 1.0]

# Others
config.fp16 = True
config.dali = False # For Large Sacle Dataset, such as WebFace42M
config.seed = 2048

# -----------------------------------------------------------------------------
# Output and Saving
# -----------------------------------------------------------------------------

config.save_all_states = True
config.output = "/home/lslopes/Swin-Transformer-Object-Detection/SwinFace/output" ####Path for Output

config.verbose = 2000
config.save_verbose = 4000
config.frequent = 10





















