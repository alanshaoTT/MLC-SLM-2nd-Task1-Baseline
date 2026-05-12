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
