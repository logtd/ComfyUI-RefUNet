# ComfyUI-RefUNet
A set of nodes to use Reference UNets


https://github.com/user-attachments/assets/5c921956-7bf2-4521-a8bf-1c8594b46641


Should be compatible with sampling methods that use reference unets, e.g.:
* [FollowYourEmoji](https://github.com/mayuelala/FollowYourEmoji)
* [MusePose](https://github.com/TMElyralab/MusePose)
* [AnimateAnyone](https://github.com/guoqincode/Open-AnimateAnyone)

## Examples
You can find examples of FollowYourEmoji in the `example_workflows` directory using @Kijai's FYE embedding nodes

https://github.com/user-attachments/assets/6b2bf9b2-8c4e-4b6b-a65d-228dc293563d

## Installation
There are no specific python requirements for this repo.

### Models
You can find the models for FollowYourEmoji here https://huggingface.co/Kijai/FollowYourEmoji-safetensors/tree/main

| Checkpoint | Directory |
|------------|-----------|
|FYE_unet-fp16.safetensors            |   unet |
|FYE_referencenet-fp16.safetensors    |   unet |
|fye_motion_module-fp16.safetensors   |  animatediff_models |
| sd-image-variations-encoder-fp16.safetensors | clip_vision |
