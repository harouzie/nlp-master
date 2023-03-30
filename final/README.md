---
license: mit
datasets:
- xsum
language:
- en
metrics:
- rouge
library_name: transformers
pipeline_tag: summarization
---

# BART (base-sized model) fine-tuned on `xsum`

**Disclaimer**: This [`bart-base`](https://huggingface.co/facebook/bart-base) model only fine-tuned on a small part of [`xsum`](https://huggingface.co/datasets/xsum) dataset. Due to lack of resources, using a P100 GPU, we trained it with different stages and data. The progress is described as below. You can train this model on more data before use it.

## Model description

BART has achieved state-of-the-art results on the CNN/Daily Mail and XSum datasets for summarization tasks.

- On the CNN/Daily Mail dataset, BART achieved a `ROUGE-2` score of 21.28, which is the highest reported score on this dataset as of September 2021. The previous state-of-the-art model, [`google/PEGASUS`](https://huggingface.co/google/pegasus-xsum), achieved a `ROUGE-2` score of 21.00. BART also achieved state-of-the-art results on several other metrics such as `ROUGE-1` and `ROUGE-L`.

- On the XSum dataset, BART achieved a `ROUGE-2` score of 22.27, which is the highest reported score on this dataset as of September 2021. The previous state-of-the-art model, T5, achieved a `ROUGE-2` score of 22.06. BART also achieved state-of-the-art results on several other metrics such as `ROUGE-1` and `ROUGE-L`.

BART SOTA on CNN/DM

```m
{
    'eval_rouge1': 44.16,
    'eval_rouge2': 21.28,
    'eval_rougeL': 40.90
}
```

BART SOTA on XSum

```m
{
    'eval_rouge1': 45.14,
    'eval_rouge2': 22.27,
    'eval_rougeL': 37.25
}
```

## Training Strategy

### **First round**

At first, we tested GPU memory with first 10k samples and batch_size of 16

Data: train/test/validation[10000:1000:1000] \
Epoch: 3

Evaluation:

```m
{
    'eval_loss': 3.34855318069458,
    'eval_rouge1': 35.1931,
    'eval_rouge2': 13.7162,
    'eval_rougeL': 28.4343,
    'eval_rougeLsum': 28.4329,
    'eval_gen_len': 19.58,
    'eval_runtime': 111.2625,
    'eval_samples_per_second': 8.988,
    'eval_steps_per_second': 2.247,
    'epoch': 3.0
}
```

### **Second round**

In the second round, we doubled everything by picking next 20k samples (no overlapping with first 10k) and the same batch_size of 16, also increase epoch to 5

Data: train/test/validation split[20000:2000:2000] \
Epoch: 5

Evaluation:

```m
{
    'eval_loss': 3.2764062881469727,
    'eval_rouge1': 36.4663,
    'eval_rouge2': 15.1419,
    'eval_rougeL': 30.0491,
    'eval_rougeLsum': 30.0254,
    'eval_gen_len': 19.619,
    'eval_runtime': 217.6418,
    'eval_samples_per_second': 9.189,
    'eval_steps_per_second': 2.297,
    'epoch': 5.0
}
```

Our draft training seems converged but has not achieved the SOTA point stated in the paper yet. Stay tuned for round 3

### **Round 3**

Data: train/test/validation split[70000:7000:7000] \
Epoch: 5

```m
{
    'eval_loss': 3.1328420639038086,
    'eval_rouge1': 37.3896,
    'eval_rouge2': 16.406,
    'eval_rougeL': 30.8594,
    'eval_rougeLsum': 30.8619,
    'eval_gen_len': 19.6073,
    'eval_runtime': 656.091,
    'eval_samples_per_second': 10.669,
    'eval_steps_per_second': 1.334,
    'epoch': 3.0
}
```

## How to use

Here is how to use and start fine-tuning this model on more data:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline

checkpoint = 'harouzie/bart-base-xsum'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# this bit of news link was cited from CNN: https://edition.cnn.com/2023/03/18/americas/ecuador-earthquake
news = """
At least 13 people died after a magnitude 6.8 earthquake struck southern Ecuador on Saturday afternoon, according to government officials.

The earthquake struck near the southern town of Baláo and was more than 65 km (nearly 41 miles) deep, according to the United States Geological Survey.

An estimated 461 people were injured in the quake, according to a report from the Ecuadorian president’s office. The government had previously reported that 16 people were killed but later revised the death toll.

In the province of El Oro, at least 11 people died. At least one other death was reported in the province of Azuay, according to the communications department for Ecuador’s president. In an earlier statement, authorities said the person in Azuay was killed when a wall collapsed onto a car and that at least three of the victims in El Oro died when a security camera tower came down.
"""

summarizer = pipeline(task="summarization", model=model, tokenizer=tokenizer)

summarizer(news)
```

> ```>>>[{'summary_text': 'At least 13 people have been killed and more than 500 injured in an earthquake in Ecuador, officials say.'}]```
