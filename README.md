# Selective Attention in 3D: Geometrically Informed Mask Selection for Self-Supervised Point Cloud Learning

The official implementation of our paper "Selective Attention in 3D: Geometrically Informed Mask Selection for Self-Supervised Point Cloud Learning".

![image](figs/Main3-1.png)

We propose a novel self-supervised learning technique for point clouds that leverages a geometrically informed masked selection strategy, SA3D, to enhance the learning efficiency of MAEs. Our approach deviates from traditional random mask selection, instead employing a teacher-student paradigm that identifies and targets complex regions within the data, thereby directing the model's focus to areas rich in geometric detail. This strategy is grounded in the hypothesis that concentrating on harder-to-predict patches yields a more robust feature representation, as evidenced by the improved performance on downstream tasks. Our method also incorporates knowledge distillation, allowing a fully observed knowledge teacher to transfer complex geometric understanding to a partially observed student. This significantly improves the student network's ability to infer and reconstruct masked point clouds. Extensive experiments confirm our method's superiority over SOTA benchmarks, demonstrating marked improvements in classification, segmentation, and few-shot tasks. 


### Requirements
- [Python 3.8](https://www.python.org/)
- [CUDA 11.8](https://developer.nvidia.com/cuda-zone)
- [PyTorch 1.13.1](https://pytorch.org/)
- [TorchVision 0.14.1](https://pytorch.org/)
- [Numpy 1.24.3](https://numpy.org/)
- [timm 0.4.5](https://github.com/rwightman/pytorch-image-models)

Other packages:
```
pip install -r requirements.txt
```
