# dataset settings
seed = 0
dataset_type = "IMAGE_CLASSIFICATION"
crop = "PEPPER"
modality = "image"  # image, text, tabular
num_classes = 10
data_root = "data/image_data/2024-08-22_chilli_top_10"
image_dir = "./data/image_data/2024-08-22_chilli_top_10"
label_name = "expert_classification"


label_sel = 1  # which label/attribute we want to diagnose
train_label_sel = label_sel  # 1 for noisy
test_label_sel = train_label_sel

file_name = "p10"
dataset_type += "_" + file_name
save_path = f"./results/{dataset_type}/"

feature_type = "embedding"

embedding_model = "hf-hub:imageomics/bioclip"
embedding_cfg = dict(
    save_num=200,
    shuffle=False,
    batch_size=128,
    num_workers=1,
)


accuracy = dict(topk=1, threth=0.5)
n_epoch = 10
print_freq = 390
details = True

train_cfg = dict(
    shuffle=True,
    batch_size=128,
    num_workers=1,
)

test_cfg = dict(
    shuffle=False,
    batch_size=128,
    num_workers=1,
)


optimizer = dict(name="SGD", config=dict(lr=0.1))


hoc_cfg = dict(
    max_step=1501,
    T0=None,
    p0=None,
    lr=0.05,
    num_rounds=50,
    sample_size=11_000,
    already_2nn=False,
    device="cpu",
)


detect_cfg = dict(
    num_epoch=51, sample_size=11_000, k=10, name="simifeat", method="rank"
)
