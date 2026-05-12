#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


def load_lora_model(base_model, lora_path, device="cuda"):
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    print(f"Loading processor from: {base_model}")
    processor = VibeVoiceASRProcessor.from_pretrained(
        base_model,
        language_model_pretrained_name="Qwen/Qwen2.5-7B",
    )

    print(f"Loading base model from: {base_model}")
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        base_model,
        dtype=dtype,
        device_map=None,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    if device != "auto":
        model = model.to(device)

    print(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    return model, processor


def read_scp(scp_file):
    items = []
    with open(scp_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                utt_id = Path(parts[0]).stem
                wav_path = parts[0]
            else:
                utt_id, wav_path = parts

            items.append((utt_id, wav_path))

    return items


def load_finished(output_file):
    finished = set()
    if not os.path.exists(output_file):
        return finished

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if "utt" in item:
                    finished.add(item["utt"])
            except Exception:
                continue

    return finished


@torch.no_grad()
def transcribe_one(
    model,
    processor,
    audio_path,
    device="cuda",
    max_new_tokens=4096,
    temperature=0.0,
    context_info=None,
):
    inputs = processor(
        audio=audio_path,
        sampling_rate=None,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        context_info=context_info,
    )

    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": processor.pad_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }

    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    output_ids = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_len:]
    raw_text = processor.decode(generated_ids, skip_special_tokens=True)

    try:
        segments = processor.post_process_transcription(raw_text)
    except Exception:
        segments = []

    return raw_text, segments


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference with LoRA fine-tuned VibeVoice-ASR."
    )

    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--scp_file", required=True)
    parser.add_argument("--output_file", required=True)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--context_info", default=None)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = read_scp(args.scp_file)
    print(f"Loaded {len(items)} utterances from {args.scp_file}")

    finished = load_finished(args.output_file) if args.resume else set()
    if finished:
        print(f"Resume enabled. Skip {len(finished)} finished utterances.")

    model, processor = load_lora_model(
        base_model=args.base_model,
        lora_path=args.lora_path,
        device=args.device,
    )

    with open(args.output_file, "a", encoding="utf-8") as fout:
        for idx, (utt, wav_path) in enumerate(items):
            if utt in finished:
                continue

            print(f"[{idx + 1}/{len(items)}] {utt}: {wav_path}")

            try:
                raw_text, segments = transcribe_one(
                    model=model,
                    processor=processor,
                    audio_path=wav_path,
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    context_info=args.context_info,
                )

                result = {
                    "utt": utt,
                    "wav": wav_path,
                    "raw_text": raw_text,
                    "segments": segments,
                }

            except Exception as e:
                result = {
                    "utt": utt,
                    "wav": wav_path,
                    "error": str(e),
                }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Done. Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
