
# AudioSR: Versatile Audio Super-resolution at Scale

[![arXiv](https://img.shields.io/badge/arXiv-2309.07314-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2309.07314)  [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://audioldm.github.io/audiosr) [![Replicate](https://replicate.com/nateraw/audio-super-resolution/badge)](https://replicate.com/nateraw/audio-super-resolution)

Pass your audio in, AudioSR will make it high fidelity! 

Work on all types of audio (e.g., music, speech, dog, raining, ...) & all sampling rates.

Share your thoughts/samples/issues in our discord channel: https://discord.gg/HWeBsJryaf

![Image Description](https://github.com/haoheliu/versatile_audio_super_resolution/blob/main/visualization.png?raw=true)

## Locally
### Gradio (WEB UI) Demo

To run the Gradio demo locally:

1. Install dependencies: `pip install -r requirements.txt` 
2. Run the app: `python app.py`
3. Open the URL displayed to view the demo

### Commandline (NO UI) Usage

#### Installation
```shell
# Optional
conda create -n audiosr python=3.9; conda activate audiosr
# Install AudioLDM (NO UI ONLY)
pip3 install audiosr
# or for using also the gradio demo you can do the following
pip3 install git+https://github.com/Nick088Official/versatile_audio_super_resolution.git
```

#### Usage

Process a list of files. The result will be saved at ./output by default.

```shell
audiosr -il batch.lst
```

Process a single audio file.
```shell
audiosr -i example/music.wav
```

Full usage instruction

```shell
> audiosr -h

> usage: audiosr [-h] -i INPUT_AUDIO_FILE [-il INPUT_FILE_LIST] [-s SAVE_PATH] [--model_name {basic,speech}] [-d DEVICE] [--ddim_steps DDIM_STEPS] [-gs GUIDANCE_SCALE] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_AUDIO_FILE, --input_audio_file INPUT_AUDIO_FILE
                        Input audio file for audio super resolution
  -il INPUT_FILE_LIST, --input_file_list INPUT_FILE_LIST
                        A file that contains all audio files that need to perform audio super resolution
  -s SAVE_PATH, --save_path SAVE_PATH
                        The path to save model output
  --model_name {basic,speech}
                        The checkpoint you gonna use
  -d DEVICE, --device DEVICE
                        The device for computation. If not specified, the script will automatically choose the device based on your environment.
  --ddim_steps DDIM_STEPS
                        The sampling step for DDIM
  -gs GUIDANCE_SCALE, --guidance_scale GUIDANCE_SCALE
                        Guidance scale (Large => better quality and relavancy to text; Small => better diversity)
  --seed SEED           Change this value (any integer number) will lead to a different generation result, also if you set it to 0, it will use a random one.
  --suffix SUFFIX       Suffix for the output file
```

## Online

### Google Colabs
- UI <a target="_blank" href="https://colab.research.google.com/github/Nick088Official/versatile_audio_super_resolution/blob/main/Audio_SR_WEB_UI.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- NO UI <a target="_blank" href="https://colab.research.google.com/github/Nick088Official/versatile_audio_super_resolution/blob/main/Audio_SR_NO_UI.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Hugging Face Space
[Hugging Face Space🤗](https://huggingface.co/spaces/Nick088/Audio-SR) 

### Kaggle
- UI (WORK IN PROGRESS)
- NO UI (WORK IN PROGRESS)

## Change Log
- 2024-07-02: Added NO UI Google Colab & Set Random seed when input seed value is 0.
- 2024-05-10: Added Gradio Demo.
- 2023-09-24: Add replicate demo (@nateraw); Fix error on windows, librosa warning etc (@ORI-Muchim).  
- 2023-09-16: Fix DC shift issue. Fix duration padding bug. Update default DDIM steps to 50.

## TODO
[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/haoheliuP)

- [ ] Optimize the inference speed.

## Cite our work
If you find this repo useful, please consider citing: 
```bibtex
@article{liu2023audiosr,
  title={{AudioSR}: Versatile Audio Super-resolution at Scale},
  author={Liu, Haohe and Chen, Ke and Tian, Qiao and Wang, Wenwu and Plumbley, Mark D},
  journal={arXiv preprint arXiv:2309.07314},
  year={2023}
}
```
