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

```
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

### Datasets

We use ShapeNet, ScanObjectNN, ModelNet40 and ShapeNetPart in this work. See [Point-MAE Repo](https://github.com/Pang-Yatian/Point-MAE/blob/main/DATASET.md) for details.


### Pre-training
Pre-trained by ShapeNet, Point-M2AE is evaluated by **Linear SVM** on ModelNet40 and ScanObjectNN (OBJ-BG split) datasets, without downstream fine-tuning:
| Model | Task | Dataset | Config | MN40 Acc.| 
| :-----: | :-----: |:-----:| :-----: | :-----: |
| Point-MAE+SA3DF | Pre-training | ShapeNet |[point-mae+SA3DF.yaml](./Point-MAE_SA3D/cfgs/config.yaml)| 92.30% |
| Point-M2AE+SA3DF | Pre-training | ShapeNet |[point-m2ae+SA3DF.yaml](./Point-M2AE_SA3D/cfgs/config_Point_M2AE.yaml)| 93.15% |

### Fine-tuning
Synthetic shape classification on ModelNet40 with 1k points:
| Models | Task  | Config | Acc.| Vote | Logs |   
| :-----: | :-----:| :-----:| :-----: | :-----:| :-----:|
| Point-MAE+SA3F | Classification | [modelnet40.yaml](./Point-MAE_SA3D/cfgs/finetune_modelnet.yaml)|93.55%| 94.16% | [modelnet40.log](./Point-MAE_SA3D/log_files/modelnet40.log) |
| Point-M2AE+SA3F | Classification | [modelnet40.yaml](./Point-M2AE_SA3D/cfgs/finetune_modelnet_PointM2AE.yaml)|92.90%| 93.03% | [modelnet40.log](./Point-M2AE_SA3D/log_files/modelnet40.txt) |

Real-world shape classification on ScanObjectNN:
| Model | Task | Split | Config | Acc. | Logs |   
| :-----: | :-----:|:-----:| :-----:| :-----:|:-----:|
| Point-MAE+SA3DF | Classification | PB-T50-RS|[scan_pb.yaml](./Point-MAE_SA3D/cfgs/config_finetune_scan_hardest.yaml) | 85.35%| [scan_pd.log](./Point-MAE_SA3D/log_files/hardest.txt) |
| Point-MAE+SA3DF | Classification |OBJ-BG| [scan_obj-bg.yaml](./Point-MAE_SA3D/cfgs/config_finetune_scan_objbg.yaml) | 90.70%| [scan_obj-pd.log](./Point-MAE_SA3D/log_files/obj_bg.txt) |
| Point-MAE+SA3DF | Classification | OBJ-ONLY| [scan_obj.yaml](./Point-MAE_SA3D/cfgs/config_finetune_scan_objonly.yaml) | 90.36%| [scan_obj.log](./Point-MAE_SA3D/log_files/obj_only.txt) |
| Point-M2AE+SA3DF | Classification | PB-T50-RS|[scan_pb.yaml](./Point-M2AE_SA3D/cfgs/config_finetune_scan_hardest_PointM2AE.yaml) | 86.47%| [scan_pd.log](./Point-M2AE_SA3D/log_files/hardest.txt) |
| Point-M2AE+SA3DF | Classification |OBJ-BG| [scan_obj-bg.yaml](./Point-M2AE_SA3D/cfgs/config_finetune_scan_objbg_PointM2AE.yaml) | 92.42%| [scan_obj-pd.log](./Point-M2AE_SA3D/log_files/obj_bg.txt) |
| Point-M2AE+SA3DF | Classification | OBJ-ONLY| [scan_obj.yaml](./Point-M2AE_SA3D/cfgs/config_finetune_scan_objonly_PointM2AE.yaml) | 89.50%| [scan_obj.log](./Point-M2AE_SA3D/log_files/obj_only.txt) |

Part segmentation on ShapeNetPart:
| Model | Task | Dataset | mIoUc| mIoUi | Logs |   
| :-----: | :-----: |:-----:| :-----:| :-----: | :-----:|
| Point-MAE+SA3DF | Segmentation | ShapeNetPart | 84.49% | 86.04% | [seg.log](./Point-MAE_SA3D/log_files/segmentation.txt) |
| Point-M2AE+SA3DF | Segmentation | ShapeNetPart |84.91% | 86.52% | - |

Few-shot classification on ModelNet40:
| Model |  Task | Dataset | Config | 5w10s | 5w20s | 10w10s| 10w20s|     
| :-----: | :-----: |:-----:| :-----: | :-----:|:-----:|:-----:| :-----:|
| Point-MAE+SA3DF |  Few-shot Cls. | ModelNet40 |[fewshot.yaml](./Point-MAE_SA3D/cfgs/fewshot.yaml) | 97.0%|98.3%|93.1%|95.2%| 


## Get Started

### Pre-training
Point-MAE+SA3DF and Point-M2AEe+SA3DF are pre-trained on ShapeNet dataset with the config files `./Point-MAE_SA3D/cfgs/config.yaml and ./Point-M2AE_SA3D/cfgs/config_Point_M2AE.yaml`. 

Run for Point-MAE+SA3DF:
```bash
CUDA_VISIBLE_DEVICES=<GPUs> python main_pretrain.py --config ./Point-MAE_SA3D/cfgs/config.yaml  --exp_name pre-train
```
Run for Point-M2AE+SA3DF:
```bash
CUDA_VISIBLE_DEVICES=<GPUs> python main_pretrain.py --config ./Point-M2AE_SA3D/cfgs/config_Point_M2AE.yaml --exp_name pre-train
```

### Fine-tuning
Please create a folder `ckpts/` and put pretrained_model in it.

For ModelNet40 (Point-MAE+SA3DF), run:
```bash
CUDA_VISIBLE_DEVICES=<GPUs> python main_finetune.py --config ./Point-MAE_SA3D/cfgs/finetune_modelnet.yaml --finetune_model --exp_name finetune --ckpts ckpts/pretrained_model.pth
```

For ModelNet40 (Point-M2AE+SA3DF), run:
```bash
CUDA_VISIBLE_DEVICES=<GPUs> python main_finetune.py --config ./Point-M2AE_SA3D/cfgs/finetune_modelnet_PointM2AE.yaml --finetune_model --exp_name finetune --ckpts ckpts/pretrained_model.pth
```

For the three splits of ScanObjectNN (Point-MAE+SA3DF), run:

```bash
CUDA_VISIBLE_DEVICES=<GPUs> python main_finetune.py --config ./Point-MAE_SA3D/cfgs/config_finetune_scan_hardest.yaml --finetune_model --exp_name finetune --ckpts ckpts/pretrained_model.pth
```
```bash
CUDA_VISIBLE_DEVICES=<GPUs> python main_finetune.py --config /Point-MAE_SA3D/cfgs/config_finetune_scan_objonly.yaml --finetune_model --exp_name finetune --ckpts ckpts/pretrained_model.pth
```
```bash
CUDA_VISIBLE_DEVICES=<GPUs> python main_finetune.py --config ./Point-MAE_SA3D/cfgs/config_finetune_scan_objbg.yaml --finetune_model --exp_name finetune --ckpts ckpts/pretrained_model.pth




