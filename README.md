# Lightweight and High-Fidelity End-to-End Text-to-Speech with Multi-Band Generation and Inverse Short-Time Fourier Transform
### Masaya Kawamura, Yuma Shirahata, Ryuichi Yamamoto, Kentaro Tachibana
We propose a lightweight end-to-end text-to-speech model using multi-band generation and inverse short-time Fourier transform. Our model is based on VITS, a high-quality end-to-end text-to-speech model, but adopts two changes for more efficient inference: 1) the most computationally expensive component is partially replaced with a simple inverse short-time Fourier transform, and 2) multi-band generation, with fixed or trainable synthesis filters, is used to generate waveforms. Unlike conventional lightweight models, which employ optimization or knowledge distillation separately to train two cascaded components, our method enjoys the full benefits of end-to-end optimization. Experimental results show that our model synthesized speech as natural as that synthesized by VITS, while achieving a real-time factor of 0.066 on an Intel Core i7 CPU, 4.1 times faster than VITS. Moreover, a smaller version of the model significantly outperformed a lightweight baseline model with respect to both naturalness and inference speed. Code and audio samples are available from [https://github.com/MasayaKawamura/MB-iSTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS).

You can check the [paper](https://arxiv.org/abs/2210.15975) and [demo page](https://masayakawamura.github.io/Demo_MB-iSTFT-VITS/).


<img src="./fig/proposed_model.png" width="100%">

## Multi-band iSTFT VITS and multi-stream iSTFT VITS 
This repository is based on **[official VITS code](https://github.com/jaywalnut310/vits.git)**.<br>
You can train the iSTFT-VITS, multi-band iSTFT VITS (MB-iSTFT-VITS), and multi-stream iSTFT VITS (MS-iSTFT-VITS) using this repository.<br>
We also provide the [pretrained models](https://drive.google.com/drive/folders/1CKSRFUHMsnOl0jxxJVCeMzyYjaM98aI2?usp=sharing).
### 1. Pre-requisites

0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
0. Download datasets
    1. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/), then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
```

### 2. Setting json file in [configs](configs)

| Model | How to set up json file in [configs](configs) | Sample of json file configuration|
| :---: | :---: | :---: |
| iSTFT-VITS | ```"istft_vits": true, ```<br>``` "upsample_rates": [8,8], ``` | ljs_istft_vits.json |
| MB-iSTFT-VITS | ```"subbands": 4,```<br>```"mb_istft_vits": true, ```<br>``` "upsample_rates": [4,4], ``` | ljs_mb_istft_vits.json |
| MS-iSTFT-VITS | ```"subbands": 4,```<br>```"ms_istft_vits": true, ```<br>``` "upsample_rates": [4,4], ``` | ljs_ms_istft_vits.json |

### 3. Training
In the case of MB-iSTFT-VITS training, run the following script
```sh
python train_latest.py -c configs/ljs_mb_istft_vits.json -m ljs_mb_istft_vits

```

After the training, you can check inference audio using [inference.ipynb](inference.ipynb)

## References
- https://github.com/jaywalnut310/vits.git
- https://github.com/rishikksh20/iSTFTNet-pytorch.git
- https://github.com/rishikksh20/melgan.git
