#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import shutil
import wave
from pathlib import Path


def get_wav_duration(wav_path: Path) -> float:
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return round(wf.getnframes() / float(wf.getframerate()), 2)
    except Exception:
        return 0.0


def parse_txt(txt_path: Path):
    segments = []

    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = re.split(r"\s+", line, maxsplit=3)
            if len(parts) < 4:
                continue

            start, end, speaker, text = parts

            try:
                start = round(float(start), 2)
                end = round(float(end), 2)
            except ValueError:
                continue

            speaker = speaker.strip().upper()
            if speaker == "O1":
                speaker_id = 0
            elif speaker == "O2":
                speaker_id = 1
            else:
                match = re.match(r"O(\d+)", speaker)
                speaker_id = int(match.group(1)) - 1 if match else -1

            if end <= start:
                continue

            segments.append({
                "speaker": speaker_id,
                "text": text.strip(),
                "start": start,
                "end": end
            })

    return segments


def convert_dataset(raw_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    counter = 0
    skipped = 0

    for set_dir in sorted(raw_dir.glob("MLC-SLM_Workshop-Training_Set_*")):
        data_dir = set_dir / "data"
        if not data_dir.exists():
            continue

        for wav_path in sorted(data_dir.rglob("*.wav")):
            txt_path = wav_path.with_suffix(".txt")

            if not txt_path.exists():
                skipped += 1
                print(f"[Skip] Missing txt: {wav_path}")
                continue

            segments = parse_txt(txt_path)
            if not segments:
                skipped += 1
                print(f"[Skip] Empty segments: {txt_path}")
                continue

            audio_name = f"{counter}.wav"
            json_name = f"{counter}.json"

            shutil.copy2(wav_path, out_dir / audio_name)

            label = {
                "audio_duration": get_wav_duration(wav_path),
                "audio_path": audio_name,
                "segments": segments
            }

            with (out_dir / json_name).open("w", encoding="utf-8") as f:
                json.dump(label, f, ensure_ascii=False, indent=2)

            counter += 1

    print("=" * 50)
    print("Done.")
    print(f"Output dir      : {out_dir}")
    print(f"Total converted : {counter}")
    print(f"Skipped         : {skipped}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MLC-SLM raw data into audio-json training format."
    )
    parser.add_argument(
        "--raw-dir",
        default="./raw_data",
        help="Raw data directory containing MLC-SLM_Workshop-Training_Set_* folders.",
    )
    parser.add_argument(
        "--out-dir",
        default="./train",
        help="Output training directory.",
    )

    args = parser.parse_args()

    convert_dataset(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
