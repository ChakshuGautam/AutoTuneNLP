from workflow.training.tasks import Tasks
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
    WhisperForConditionalGeneration, Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from functools import partial
import evaluate
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperFineTuning(Tasks):
    def __init__(self, model_name: str, version: str, args):
        super().__init__("whisper_finetuning", model_name, version)
        self.metrics = evaluate.load("wer")
        self.processor = WhisperProcessor.from_pretrained(model_name, language=args["language"], task="transcribe")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.args = args

    def process_and_upload_dataset(self, dataset,dataset_name):
        import tempfile
        import os
        from gtts import gTTS
        from datasets import load_dataset, DatasetDict, Audio, Dataset
        from huggingface_hub import HfApi, create_repo
        import random
        temp_dir = tempfile.mkdtemp()
        import os
        import azure.cognitiveservices.speech as speechsdk
        speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('AZURE_TTS_KEY'),region = os.environ.get('AZURE_TTS_REGION'))
        speech_config.speech_synthesis_voice_name='en-US-AvaMultilingualNeural'

        def text_to_audio(text):
            audio_path = os.path.join(temp_dir, f"audio_{hash(text)}.wav")
            audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_path)
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            result = speech_synthesizer.speak_text_async(text).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return audio_path
            else:
                print(f"Error synthesizing audio: {result.reason}")
                return None

        def process_split(split):
            audio_data = []
            sentences = []
            remaining = ""
            for example in split:
                text = example[split.column_names[0]]
                if len(text) > 200:
                    remaining+=(text+" ")
                    continue
                audio_path = text_to_audio(text)
                audio_data.append({"path": audio_path})
                sentences.append(text)
            if remaining:
                while len(remaining) > 200:
                    split_index = remaining[:200].rfind(' ')
                    if split_index == -1:
                        split_index = 200
                    text = remaining[:split_index].strip()
                    audio_path = text_to_audio(text)
                    audio_data.append({"path": audio_path})
                    sentences.append(text)
                    remaining = remaining[split_index:].strip()
            if remaining:
                audio_path = text_to_audio(remaining)
                audio_data.append({"path": audio_path})
                sentences.append(remaining)
            processed_split = Dataset.from_dict({
                "audio": audio_data,
                "sentence": sentences
            })
            processed_split = processed_split.cast_column("audio", Audio())
            return processed_split

        def create_test_split(dataset, test_size=0.2):
            data = list(dataset)
            random.shuffle(data)
            split_index = int(len(data) * (1 - test_size))
            train_data = data[:split_index]
            test_data = data[split_index:]
            return Dataset.from_list(train_data), Dataset.from_list(test_data)

        train_dataset = process_split(dataset['train'])
        if 'test' in dataset:
            test_dataset = process_split(dataset['test'])
        else:
            train_dataset, test_dataset = create_test_split(train_dataset)

        processed_dataset = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        api = HfApi()
        repo_id = f"{dataset_name}__audio"
        
        try:
            create_repo(repo_id, repo_type="dataset", token=os.environ.get('HUGGING_FACE_TOKEN'))
        except Exception as e:
            print(f"Repo already exists or couldn't be created: {e}")
        
        processed_dataset.push_to_hub(repo_id, token=os.environ.get('HUGGING_FACE_TOKEN'))
        
        print(f"Dataset uploaded successfully to {repo_id}")
        
        return processed_dataset
    
    def load_dataset(self, dataset_name):
        dataset = load_dataset(dataset_name)
        train_dataset = dataset["train"]
        if len(train_dataset.column_names) == 1:
            dataset = self.process_and_upload_dataset(dataset,dataset_name)
            train_dataset = dataset["train"]
            print("Dataset splits:", dataset.keys())
            for split_name, split_data in dataset.items():
                print(f"\nSplit: {split_name}")
                print("Columns:", split_data.column_names)
                print("Number of rows:", len(split_data))
                
        if "audio" not in train_dataset.column_names:
            audio_column = train_dataset.column_names[0]
            train_dataset = train_dataset.rename_column(audio_column, "audio")

        if "sentence" not in train_dataset.column_names:
            sentence_column = train_dataset.column_names[1]
            train_dataset = train_dataset.rename_column(sentence_column, "sentence")
        
        train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
        dataset = train_dataset.train_test_split(test_size=0.1)
        self.train_dataset = dataset["train"]
        self.eval_dataset = dataset["test"]
        self._load_model()
        return dataset
    
    def _prepare_dataset(self):
        pass

    def _load_model_requirements(self):
        pass

    def _load_model(self):
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        self.model.config.dropout = 0.1

        self.Trainer = partial(
            Seq2SeqTrainer,
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            data_collator=self.DataCollatorSpeechSeq2SeqWithPadding(
                feature_extractor=self.feature_extractor,
                tokenizer=self.tokenizer,
                processor=self.processor,
                decoder_start_token_id=self.model.config.decoder_start_token_id
            )
        )

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        feature_extractor: Any
        tokenizer: Any
        processor: Any
        decoder_start_token_id: int

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": self.feature_extractor(f["audio"]["array"], sampling_rate=f["audio"]["sampling_rate"]).input_features[0]} for f in features]
            labels = [{"input_ids": self.tokenizer(f["sentence"]).input_ids} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            labels_batch = self.processor.tokenizer.pad(labels, return_tensors="pt")
            labels_batch["input_ids"] = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            if (labels_batch["input_ids"][:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels_batch["input_ids"] = labels_batch["input_ids"][:, 1:]

            batch["labels"] = labels_batch["input_ids"]
            return batch

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * self.metrics.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    def get_training_args(self, req_data, dataset):
        return Seq2SeqTrainingArguments(
            output_dir=f"./results_{req_data['task_id']}",
            num_train_epochs=req_data["epochs"],
            per_device_train_batch_size=self.args.get("per_device_train_batch_size", 16),
            per_device_eval_batch_size=self.args.get("per_device_eval_batch_size", 8),
            warmup_ratio=self.args.get("warmup_ratio", 0.1),
            gradient_accumulation_steps=self.args.get("gradient_accumulation_steps", 2),
            learning_rate=self.args.get("learning_rate", 3.75e-5),
            fp16=True,
            evaluation_strategy="steps",
            eval_steps=self.args.get("eval_steps", 50),
            logging_steps=self.args.get("logging_steps", 25),
            save_steps=self.args.get("save_steps", 50),
            predict_with_generate=True,
            generation_max_length=225,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            remove_unused_columns=False,
            lr_scheduler_type=self.args.get("lr_scheduler_type", "constant"),
        )

    def push_to_hub(self, trainer, save_path, hf_token=None):
        trainer.model.push_to_hub(
            save_path, commit_message="pytorch_model.bin upload/update"
        )
        self.processor.push_to_hub(save_path, token=hf_token)
