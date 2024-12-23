from workflow.training.tasks import Tasks
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from functools import partial
import numpy as np
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, CommitOperationAdd
import io 
import json
# needs train and validation in the dataset
# needs 'class'/'label' column in the dataset
class TextClassification(Tasks):
    def __init__(self, model_name: str, version: str, args):
        super().__init__("text_classification", model_name, version)
        self.metrics = evaluate.load("f1")
        self.le = LabelEncoder()
        self.label2id = None
        self.id2label = None
        if "label2id" in args and len(args["label2id"]) != 0:
            self.label2id = args["label2id"]

    def load_dataset(self, dataset):
        self.dataset = load_dataset(dataset).shuffle()
        self._prepare_dataset()
        self._load_model()
        return self.dataset

    def _load_model(self):
        num_labels = len(self.id2label)
        print(
            f"Number of labels: {sorted(self.dataset['test'].unique('class'))} {sorted(self.dataset['train'].unique('class'))}"
        )
        assert sorted(self.dataset["test"].unique("class")) == sorted(
            self.dataset["train"].unique("class")
        ), "Train and Test dataset must have the same number of classes"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.Trainer = partial(
            Trainer,
            model=self.model,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

    def _load_model_requirements(self):
        self.Trainer = Trainer
        self.TrainingArguments = TrainingArguments

    def __label_encoder(self, examples):
        if self.label2id is not None:
            encoded_labels = np.array(
                [self.label2id[label] for label in examples["class"]]
            )

        else:
            encoded_labels = self.le.fit_transform(np.array(examples["class"]))
        return {"text": examples["text"], "label": encoded_labels}

    def _prepare_dataset(self):
        # assume label column is 'class' and text column is 'text' in the dataset
        self.dataset = self.dataset.map(self.__label_encoder, batched=True)
        self.tokenized_dataset = self.dataset.map(
            self.__preprocess_function, batched=True
        )

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        if self.label2id is None:
            self.label2id = dict(
                zip(self.le.classes_, map(str, self.le.transform(self.le.classes_)))
            )
        self.id2label = {v: k for k, v in self.label2id.items()}

    def __preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding=True)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metrics.compute(
            predictions=predictions, references=labels, average="macro"
        )

    def push_to_hub(self, trainer, save_path, hf_token=None, metrics=None, dataset_name= ''):
        trainer.model.push_to_hub(
            save_path, commit_message="pytorch_model.bin upload/update"
        )
        trainer.tokenizer.push_to_hub(save_path)

        model_card = self._build_model_card_text(save_path, metrics, dataset_name)

        ##Upload the README.md
        hf_api = HfApi(token=hf_token)
        readme_bytes = model_card.encode("utf-8")

        commit_info = hf_api.create_commit(
            repo_id=save_path,
            repo_type="model",
            commit_message="Add or update model card README",
            operations=[CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=io.BytesIO(readme_bytes))],
        )
        print(f"Model card commit info: {commit_info}")

    def _build_model_card_text(self, repo_id, metrics, dataset_name):
        """
        Returns a string that will be placed into README.md.
        You can customize this as you like.
        """
   
        md_table =  """
        | Metric | Value |
        | --- | --- | 
        """
    # Loop over the metrics to append rows
        for k, v in metrics.items():
            md_table += f"| {k} | {v} | \n "


        model_card = f"""
        license: apache-2.0
        ---
        # {repo_id}

        This is a text-classification model fine-tuned on `{self.model_name}` using the dataset `{dataset_name}`.

        ## Evaluation Results

        Below are the evaluation metrics from the training run:

        {md_table}
        """
        return model_card

    def get_training_args(self, req_data, dataset):

        return TrainingArguments(
            output_dir=f"./results_{req_data['task_id']}",
            num_train_epochs=req_data["epochs"],
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            logging_dir=f"./logs_{req_data['task_id']}",
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            warmup_steps=500,
            weight_decay=0.01,
            do_predict=True,
        )
