import torch
import utils
import argparse

from models import SynthesizerTrn
from text.symbols import symbols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model checkpoints")
    parser.add_argument(
        "--config-path", required=True, help="Path to model config (.json)"
    )
    parser.add_argument(
        "--output-onnx-path", required=True, help="Path to write exported onnx model"
    )
    parser.add_argument("--device", required=True, type=str, help="Device to run onnx export", default="cpu")
    args = parser.parse_args()


    hps = utils.get_hparams_from_file(args.config_path)

    if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder == True:
        print("Using mel posterior encoder for VITS2")
        posterior_channels = 80  # vits2
        hps.data.use_mel_posterior_encoder = True
    else:
        print("Using lin posterior encoder for VITS1")
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        # n_speakers=hps.data.n_speakers, #- for multi speaker
        is_onnx=True,
        **hps.model).to(args.device)

    _ = utils.load_checkpoint(args.model, net_g, None)

    num_symbols = net_g.n_vocab
    
    def infer_forward(text, text_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio = net_g.infer(
                text,
                text_lengths,
                noise_scale=noise_scale,
                length_scale=length_scale,
                noise_scale_w=noise_scale_w,
                sid=sid,
        )[0].unsqueeze(1)

        return audio

    with torch.no_grad():
        net_g.dec.remove_weight_norm()
        net_g.forward = infer_forward

    net_g.eval()

    dummy_input_length = 50
    sequences = torch.randint(
        low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long
    ).to(args.device)
    sequence_lengths = torch.LongTensor([sequences.size(1)]).to(args.device)
    sid = None

    # noise, noise_w, length
    scales = torch.FloatTensor([0.667, 1.0, 0.8]).to(args.device)
    dummy_input = (sequences, sequence_lengths, scales, sid)

    # Export
    torch.onnx.export(
            model=net_g,
            args=dummy_input,
            f=args.output_onnx_path,
            verbose=True,
            opset_version=14,
            input_names=["input", "input_lengths", "scales", "sid"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "phonemes"},
                "input_lengths": {0: "batch_size"},
                "output": {0: "batch_size", 1: "time"},
            },
    )

if __name__ == "__main__":
    main()