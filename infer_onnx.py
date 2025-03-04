import argparse

import numpy as np
import onnxruntime
import torch
import soundfile as sf
    
import commons
import utils
from text import text_to_sequence


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model (.onnx)")
    parser.add_argument(
        "--config-path", required=True, help="Path to model config (.json)"
    )
    parser.add_argument(
        "--output-wav-path", required=True, help="Path to write WAV file"
    )
    parser.add_argument("--text", required=True, type=str, help="Text to synthesize")
    parser.add_argument("--device", required=True, type=str, help="Device to run onnx export", default="cpu")
    args = parser.parse_args()

    sess_options = onnxruntime.SessionOptions()
    
    providers = ["CPUExecutionProvider"]
    if args.device == "cuda":
        providers.append("CUDAExecutionProvider")

    model = onnxruntime.InferenceSession(str(args.model), sess_options=sess_options, providers=providers)

    hps = utils.get_hparams_from_file(args.config_path)

    phoneme_ids = get_text(args.text, hps)
    text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
    text_lengths = np.array([text.shape[1]], dtype=np.int64)
    scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)

    audio = model.run(
        None,
        {
            "input": text,
            "input_lengths": text_lengths,
            "scales": scales,
            "sid": None,
        },
    )[0].squeeze((0, 1))
    
    sf.write(args.output_wav_path, audio, hps.data.sampling_rate)
    

if __name__ == "__main__":
    main()