import os
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# 모델 build
def build(config):

    tokenizer = RobertaTokenizerFast.from_pretrained(
                                        os.path.join(config.save_directory),
                                        max_len=config.max_length
                                        )

    model_config = RobertaConfig(
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_length,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.num_hidden_layers,
        type_vocab_size=1
    )

    model = RobertaForMaskedLM(config=model_config)
    print("the number of parameters of model: ", model.num_parameters())

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=config.files,
        block_size=32
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config.mlm_probability
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(config.save_directory),
        overwrite_output_dir=config.overwrite_output_dir,
        num_train_epochs=config.num_train_epochs,
        per_gpu_train_batch_size=config.per_gpu_train_batch_size,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=config.prediction_loss_only
    )

    return trainer