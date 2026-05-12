# MLC-SLM-2nd-Task1-Baseline

For Task 1, we fine-tune the challenge training set on Microsoft's open-source model VibeVoice-ASR. We use the Meeteval toolkit to compute tcpMER, where CER is applied for Japanese, Korean, and Thai, while WER is used for all other languages.

## Setup

* Clone the repo
```
git clone https://github.com/alanshaoTT/MLC-SLM-2nd-Task1-Baseline
cd MLC-SLM-2nd-Task1-Baseline
```

* Install dependency 

```bash
pip install -e .
pip install peft
pip install meeteval
```

* Data preparation

Place all downloaded training zip files under the `finetuning-asr` directory.

```bash
cd finetuning-asr
bash run.sh --stage 0 --stop-stage 1
```

This will extract the raw data and convert it into the audio-json training format.

## Model Training
We fine-tune VibeVoice-ASR with LoRA on the converted challenge training set.

```bash
bash run.sh --stage 2 --stop-stage 2
```

After training, run batch inference with the fine-tuned LoRA checkpoint:

```bash
bash run.sh --stage 3 --stop-stage 3
```

## Evaluation 
We evaluate the baseline on the official development set using tcpWER/tcpCER with collar = 5. Japanese, Korean, and Thai are evaluated using tcpCER, while all other languages use tcpWER. Avg denotes tcpMER averaged across all evaluated languages.

| Language ID | Metric |
|---|---:|
| Eng_American | 77.39 |
| Eng_Australian | 81.50 |
| Eng_British | 67.60 |
| Eng_Filipino | 63.36 |
| Eng_Indian | 72.12 |
| French | 83.39 |
| French_Canada | 78.56 |
| German | 84.23 |
| Italian | 78.16 |
| Japanese | 81.46 |
| Korean | 81.33 |
| Portuguese | 75.64 |
| Portuguese_Brazil | 73.02 |
| Russian | 83.84 |
| Spanish | 82.51 |
| Spanish_Mexico | 78.81 |
| Tagalog | 81.09 |
| Thai | 83.67 |
| Turkish | 92.97 |
| Urdu | 89.63 |
| Vietnamese | 71.81 |
| Avg. | 79.15 |
