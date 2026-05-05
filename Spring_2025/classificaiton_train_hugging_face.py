# %% [markdown]
# # **Fine-tuning for Image Classification with 🤗 Transformers**
# 
# This notebook shows how to fine-tune any pretrained Vision model for Image Classification on a custom dataset. The idea is to add a randomly initialized classification head on top of a pre-trained encoder, and fine-tune the model altogether on a labeled dataset.
# 
# ## ImageFolder
# 
# This notebook leverages the [ImageFolder](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder) feature to easily run the notebook on a custom dataset (namely, [EuroSAT](https://github.com/phelber/EuroSAT) in this tutorial). You can either load a `Dataset` from local folders or from local/remote files, like zip or tar.
# 
# ## Any model
# 
# This notebook is built to run on any image classification dataset with any vision model checkpoint from the [Model Hub](https://huggingface.co/) as long as that model has a version with a Image Classification head, such as:
# * [ViT](https://huggingface.co/docs/transformers/model_doc/vit#transformers.ViTForImageClassification)
# * [Swin Transformer](https://huggingface.co/docs/transformers/model_doc/swin#transformers.SwinForImageClassification)
# * [ConvNeXT](https://huggingface.co/docs/transformers/master/en/model_doc/convnext#transformers.ConvNextForImageClassification)
# 
# - in short, any model supported by [AutoModelForImageClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForImageClassification).
# 
# ## Data augmentation
# 
# This notebook leverages Torchvision's [transforms](https://pytorch.org/vision/stable/transforms.html) for applying data augmentation - note that we do provide alternative notebooks which leverage other libraries, including:
# 
# * [Albumentations](https://github.com/huggingface/notebooks/blob/main/examples/image_classification_albumentations.ipynb)
# * [Kornia](https://github.com/huggingface/notebooks/blob/main/examples/image_classification_kornia.ipynb)
# * [imgaug](https://github.com/huggingface/notebooks/blob/main/examples/image_classification_imgaug.ipynb).
# 
# ---
# 
# Depending on the model and the GPU you are using, you might need to adjust the batch size to avoid out-of-memory errors. Set those two parameters, then the rest of the notebook should run smoothly.
# 
# In this notebook, we'll fine-tune from the https://huggingface.co/microsoft/swin-tiny-patch4-window7-224 checkpoint, but note that there are many, many more available on the [hub](https://huggingface.co/models?other=vision).

# %%
model_checkpoint = "microsoft/swin-base-patch4-window12-384" # pre-trained model from which to fine-tune
batch_size = 16 # batch size for training and evaluation (reduced for memory)

# %% [markdown]
# Before we start, let's install the `datasets`, `transformers` and `accelerate` libraries.

# Dependencies are already installed 

# %% [markdown]
# ## Fine-tuning a model on an image classification task

# %% [markdown]
# In this notebook, we will see how to fine-tune one of the [🤗 Transformers](https://github.com/huggingface/transformers) vision models on an Image Classification dataset.
# 
# Given an image, the goal is to predict an appropriate class for it, like "tiger". The screenshot below is taken from a [ViT fine-tuned on ImageNet-1k](https://huggingface.co/google/vit-base-patch16-224) - try out the inference widget!

# %% [markdown]
# <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tiger_image.png" alt="drawing" width="600"/>
# 

# %% [markdown]
# ### Loading the dataset

# %% [markdown]
# We will use the [🤗 Datasets](https://github.com/huggingface/datasets) library's [ImageFolder](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder) feature to download our custom dataset into a DatasetDict.
# 
# In this case, the EuroSAT dataset is hosted remotely, so we provide the `data_files` argument. Alternatively, if you have local folders with images, you can load them using the `data_dir` argument.

# Dataset path: ../Dataset/yolo_detected_anole_classification


# %%
from datasets import load_dataset

# load a custom dataset from local/remote files or folders using the ImageFolder feature

# option 1: local/remote files (supporting the following formats: tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset("../Dataset/yolo_detected_anole_classification")

# %% [markdown]
# Let us also load the Accuracy metric, which we'll use to evaluate our model both during and after training.

# %%
import evaluate
metric = evaluate.load("accuracy")

# %% [markdown]
# The `dataset` object itself is a [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict), which contains one key per split (in this case, only "train" for a training split).

# %%
dataset

# %% [markdown]
# To access an actual element, you need to select a split first, then give an index:

# %%
example = dataset["train"][10]
example

# %% [markdown]
# Each example consists of an image and a corresponding label. We can also verify this by checking the features of the dataset:

# %%
dataset["train"].features

# %% [markdown]
# The cool thing is that we can directly view the image (as the 'image' field is an [Image feature](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Image)), as follows:

# %%
example['image']

# %% [markdown]
# Let's make it a little bigger as the images in the EuroSAT dataset are of low resolution (64x64 pixels):

# %%
example['image'].resize((200, 200))

# %% [markdown]
# Let's print the corresponding label:

# %%
example['label']

# %% [markdown]
# As you can see, the `label` field is not an actual string label. By default the `ClassLabel` fields are encoded into integers for convenience:

# %%
dataset["train"].features["label"]

# %% [markdown]
# Let's create an `id2label` dictionary to decode them back to strings and see what they are. The inverse `label2id` will be useful too, when we load the model later.

# %%
labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

id2label[2]

# %% [markdown]
# ### Preprocessing the data

# %% [markdown]
# Before we can feed these images to our model, we need to preprocess them.
# 
# Preprocessing images typically comes down to (1) resizing them to a particular size (2) normalizing the color channels (R,G,B) using a mean and standard deviation. These are referred to as **image transformations**.
# 
# In addition, one typically performs what is called **data augmentation** during training (like random cropping and flipping) to make the model more robust and achieve higher accuracy. Data augmentation is also a great technique to increase the size of the training data.
# 
# We will use `torchvision.transforms` for the image transformations/data augmentation in this tutorial, but note that one can use any other package (like [albumentations](https://albumentations.ai/), [imgaug](https://github.com/aleju/imgaug), [Kornia](https://kornia.readthedocs.io/en/latest/) etc.).
# 
# To make sure we (1) resize to the appropriate size (2) use the appropriate image mean and standard deviation for the model architecture we are going to use, we instantiate what is called an image processor with the `AutoImageProcessor.from_pretrained` method.
# 
# This image processor is a minimal preprocessor that can be used to prepare images for inference.

# %%
from transformers import AutoImageProcessor

image_processor  = AutoImageProcessor.from_pretrained(model_checkpoint, use_fast=True)
image_processor

# %% [markdown]
# The Datasets library is made for processing data very easily. We can write custom functions, which can then be applied on an entire dataset (either using [`.map()`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=map#datasets.Dataset.map) or [`.set_transform()`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=set_transform#datasets.Dataset.set_transform)).
# 
# Here we define 2 separate functions, one for training (which includes data augmentation) and one for validation (which only includes resizing, center cropping and normalizing).

# %%
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

# %% [markdown]
# Next, we can preprocess our dataset by applying these functions. We will use the `set_transform` functionality, which allows to apply the functions above on-the-fly (meaning that they will only be applied when the images are loaded in RAM).

# %%
# split up training into training + validation
train_ds = dataset['train']
val_ds = dataset['validation']

# %%
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

# %% [markdown]
# Let's access an element to see that we've added a "pixel_values" feature:

# %%
train_ds[0]

# %% [markdown]
# ### Training the model

# %% [markdown]
# Now that our data is ready, we can download the pretrained model and fine-tune it. For classification we use the `AutoModelForImageClassification` class. Calling the `from_pretrained` method on it will download and cache the weights for us. As the label ids and the number of labels are dataset dependent, we pass `label2id`, and `id2label` alongside the `model_checkpoint` here. This will make sure a custom classification head will be created (with a custom number of output neurons).
# 
# NOTE: in case you're planning to fine-tune an already fine-tuned checkpoint, like [facebook/convnext-tiny-224](https://huggingface.co/facebook/convnext-tiny-224) (which has already been fine-tuned on ImageNet-1k), then you need to provide the additional argument `ignore_mismatched_sizes=True` to the `from_pretrained` method. This will make sure the output head (with 1000 output neurons) is thrown away and replaced by a new, randomly initialized classification head that includes a custom number of output neurons. You don't need to specify this argument in case the pre-trained model doesn't include a head.

# %%
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)


# %% [markdown]
# The warning is telling us we are throwing away some weights (the weights and bias of the `classifier` layer) and randomly initializing some other (the weights and bias of a new `classifier` layer). This is expected in this case, because we are adding a new head for which we don't have pretrained weights, so the library warns us we should fine-tune this model before using it for inference, which is exactly what we are going to do.

# %% [markdown]
# To instantiate a `Trainer`, we will need to define the training configuration and the evaluation metric. The most important is the [`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments), which is a class that contains all the attributes to customize the training. It requires one folder name, which will be used to save the checkpoints of the model.
# 
# Most of the training arguments are pretty self-explanatory, but one that is quite important here is `remove_unused_columns=False`. This one will drop any features not used by the model's call function. By default it's `True` because usually it's ideal to drop unused feature columns, making it easier to unpack inputs into the model's call function. But, in our case, we need the unused features ('image' in particular) in order to create 'pixel_values'.

# %%
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-lizard-class-swin-base",
    remove_unused_columns=False,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=8,  # Increased to maintain effective batch size
    per_device_eval_batch_size=batch_size,
    num_train_epochs=30,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    dataloader_pin_memory=False,  # Disable pin_memory for MPS
    dataloader_num_workers=0,     # Use single worker for MPS compatibility
)

# %% [markdown]
# Here we set the evaluation to be done at the end of each epoch, tweak the learning rate, use the `batch_size` defined at the top of the notebook and customize the number of epochs for training, as well as the weight decay. Since the best model might not be the one at the end of training, we ask the `Trainer` to load the best model it saved (according to `metric_name`) at the end of training.
# 
# The last argument `push_to_hub` allows the Trainer to push the model to the [Hub](https://huggingface.co/models) regularly during training. Remove it if you didn't follow the installation steps at the top of the notebook. If you want to save your model locally with a name that is different from the name of the repository, or if you want to push your model under an organization and not your name space, use the `hub_model_id` argument to set the repo name (it needs to be the full name, including your namespace: for instance `"nielsr/vit-finetuned-cifar10"` or `"huggingface/nielsr/vit-finetuned-cifar10"`).

# %% [markdown]
# Next, we need to define a function for how to compute the metrics from the predictions, which will just use the `metric` we loaded earlier. The only preprocessing we have to do is to take the argmax of our predicted logits:

# %%
import numpy as np

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# %% [markdown]
# We also define a `collate_fn`, which will be used to batch examples together.
# Each batch consists of 2 keys, namely `pixel_values` and `labels`.

# %%
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# %% [markdown]
# Then we just need to pass all of this along with our datasets to the `Trainer`:

# %%
trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

# GPU info (nvidia-smi equivalent for monitoring)

# %% [markdown]
# You might wonder why we pass along the `image_processor` as a tokenizer when we already preprocessed our data. This is only to make sure the image processor configuration file (stored as JSON) will also be uploaded to the repo on the hub.

# %% [markdown]
# Now we can finetune our model by calling the `train` method:

# %%
train_results = trainer.train()
# rest is optional but nice to have
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

# %% [markdown]
# We can check with the `evaluate` method that our `Trainer` did reload the best model properly (if it was not the last one):

# %%
metrics = trainer.evaluate()
# some nice to haves:
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

# %% [markdown]
# Looks like our model got it correct!

# %% [markdown]
# ## Validate on Test Dataset
# 

# %%
from transformers import pipeline

pipe = pipeline("image-classification", "swin-base-patch4-window12-384-finetuned-lizard-class-swin-base")

# %%
# Get test dataset example
test_dataset = dataset["test"]

# %%
test_dataset.features

# %% [markdown]
# ### Example: Classification on 1 image

# %%
image = test_dataset[0]
image["image"].resize((200, 200))

# %%
target_image = image["image"]

# %%
pipe(target_image)

# %% [markdown]
# ### Classification on test dataset

# %%
from evaluate import evaluator

task_evaluator = evaluator("image-classification")

eval_results = task_evaluator.compute(
    model_or_pipeline=pipe,
    data=test_dataset,
    metric= "accuracy",
    label_mapping=pipe.model.config.label2id
)

eval_results

# %% [markdown]
# ### Evaluation with Confusion Matrix

# %%
# Get class names mapping
label_names = test_dataset.features["label"].names

# Create a mapping from label names to indices
label_to_idx = {name: idx for idx, name in enumerate(label_names)}

# Prepare predictions and references
def predict_image(image):
    preds = pipe(image)
    name = preds[0]["label"]
    idx = label_to_idx[name]
    return idx  # Get the top prediction

# Get predicted labels
predictions_int = [predict_image(item["image"]) for item in test_dataset]

# %%
# Get ground truth labels
references_int = [item["label"] for item in test_dataset]

# %%
#Check predictions / ground truth labels
references_int

# %%
import evaluate 

# Compute confusion matrix
conf_matrix = evaluate.load("confusion_matrix")
results = conf_matrix.compute(predictions=predictions_int, references=references_int)

# Print confusion matrix results
print(results)

# %% [markdown]
# ### Compute f1-score, precision and recall for each Anole class

# %%
# Compute precision, recall, and F1-score
metric = evaluate.combine(["precision", "recall", 'f1'])
prf_results = metric.compute(predictions=predictions_int, references=references_int, average=None)  # No averaging, get per-class metrics

# Print per-class precision, recall, and F1-score
for i, class_name in enumerate(label_names):
    print(f"Class: {class_name}")
    print(f"  Precision: {prf_results['precision'][i]:.4f}")
    print(f"  Recall: {prf_results['recall'][i]:.4f}")
    print(f"  F1-score: {prf_results['f1'][i]:.4f}")
    print("-" * 30)


# %% [markdown]
# ### Visualize Confusion Matrix (Absolute Numbers)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Convert to numpy array
conf_matrix = np.array(results["confusion_matrix"])

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=test_dataset.features["label"].names, yticklabels=test_dataset.features["label"].names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# %% [markdown]
# ### Visualize Confusion Matrix (Normalized)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Convert to numpy array
conf_matrix = np.array(results["confusion_matrix"])

conf_matrix_normalized = conf_matrix.astype("float") / conf_matrix.sum(axis=1, keepdims=True)

# Ensure NaNs (if any row sums to 0) are replaced with 0
conf_matrix_normalized = np.nan_to_num(conf_matrix_normalized)

# Plot the normalized confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix_normalized * 100,  # Convert to percentage
    annot=True, fmt=".2f", cmap="Blues",
    xticklabels=label_names, yticklabels=label_names
)

plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Normalized Confusion Matrix")
plt.show()


