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
Time-Constrained minimum-Permutation Word Error Rate (tcpWER) or Character Error Rate (tcpCER) with collar = 5

| Language ID | Metric |
|---|---:|
| eng_american | 77.39 |
| eng_australian | 81.50 |
| eng_british | 67.60 |
| eng_filipino | 63.36 |
| eng_indian | 72.12 |
| french | 83.39 |
| french_canada | 78.56 |
| german | 84.23 |
| italian | 78.16 |
| japanese | 81.46 |
| korean | 81.33 |
| portuguese | 75.64 |
| portuguese_brazil | 73.02 |
| russian | 83.84 |
| spanish | 82.51 |
| spanish_mexico | 78.81 |
| tagalog | 81.09 |
| thai | 83.67 |
| turkish | 92.97 |
| urdu | 89.63 |
| vietnamese | 71.81 |
