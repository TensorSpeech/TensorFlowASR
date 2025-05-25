import os

import datasets
import librosa
import soundfile
from tqdm import tqdm

from tensorflow_asr.utils import cli_util, data_util

MAPPING = {
    "audio.array": "audio",
    "audio.sampling_rate": "sample_rate",
    "transcription": "transcript",
}


def load_item_from_mapping(item):
    data = {}
    for path, key in MAPPING.items():
        data[key] = data_util.get(item, path)
    if not all(x in data for x in ["audio", "transcript"]):
        return None
    return data["audio"], int(data["sample_rate"]), str(data["transcript"])


def main(
    directory: str,
    token: str,
):
    dataset_list = datasets.load_dataset("linhtran92/viet_bud500", token=token, streaming=True, keep_in_memory=False)
    for stage in dataset_list.keys():
        print(f"[Loading {stage}]")
        output = os.path.realpath(os.path.join(directory, stage, "audio"))
        tsv_output = os.path.realpath(os.path.join(directory, stage, "transcripts.tsv"))
        os.makedirs(output, exist_ok=True)
        with open(tsv_output, "w", encoding="utf-8") as out:
            out.write("PATH\tDURATION\tTRANSCRIPT\n")
            index = 1
            for item in tqdm(dataset_list[stage], desc=f"[Loading to {output}]", disable=False):
                data = load_item_from_mapping(item)
                if data is None:
                    continue
                audio, sample_rate, transcript = data
                path = os.path.join(output, f"{index}.wav")
                soundfile.write(path, audio, sample_rate)
                duration = librosa.get_duration(y=audio, sr=sample_rate)
                out.write(f"{path}\t{duration}\t{transcript}\n")
                index += 1


if __name__ == "__main__":
    cli_util.run(main)
