# ByteDance/SDXL-Lightning Cog Model

This is an implementation of [ByteDance/SDXL-Lightning 4step Unet](https://huggingface.co/ByteDance/SDXL-Lightning) as a [Cog](https://github.com/replicate/cog) model.

[![Try a demo on Replicate](https://img.shields.io/static/v1?label=Demo&message=Replicate&color=blue)](https://replicate.com/bytedance/sdxl-lightning-4step)

- [x] Cog Fast Push Compatible

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of SDXL to [Replicate](https://replicate.com).

## Basic Usage

To run a prediction:

    cog predict -i prompt="A girl smiling"

# Output

![output](output.0.png)
