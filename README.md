# pytorch-superpoint

这是 "SuperPoint: Self-Supervised Interest Point Detection and Description." 的Pytorh实现 Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich. [ArXiv 2018](https://arxiv.org/abs/1712.07629).
此代码部分基于tensorflow实现
https://github.com/rpautrat/SuperPoint.

如果这有助于你的研究，请star这个repo。
本repo是我们paper的bi-product [deepFEPE(IROS 2020)](https://github.com/eric-yyjau/pytorch-deepFEPE.git).

## 我们的实现与原始文件之间的差异
- *Descriptor loss*: 我们使用不同的方法描述loss，包括密集方法（如paper所示，但略有不同）和稀疏方法。我们注意到稀疏损失可以更有效地收敛到类似地性能，这里的默认设置是稀疏方法.

## HPatches上的结果
| 任务                                       | Homography estimation |      |      | Detector metric |      | Descriptor metric |                |
|-------------------------------------------|-----------------------|------|------|-----------------|------|-------------------|----------------|
|                                           | Epsilon = 1           | 3    | 5    | Repeatability   | MLE  | NN mAP            | Matching Score |
| 预训练模型                                  | 0.44                  | 0.77 | 0.83 | 0.606           | 1.14 | 0.81              | 0.55           |
| Sift (亚像素精度)                           | 0.63                  | 0.76 | 0.79 | 0.51            | 1.16 | 0.70               | 0.27            |
| superpoint_coco_heat2_0_170k_hpatches_sub | 0.46                  | 0.75 | 0.81 | 0.63            | 1.07 | 0.78              | 0.42           |
| superpoint_kitti_heat2_0_50k_hpatches_sub | 0.44                  | 0.71 | 0.77 | 0.56            | 0.95 | 0.78              | 0.41           |

- 预训练模型来自 [SuperPointPretrainedNetwork](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork).
- 评估是在我们的评估脚本下完成的
- COCO/ KITTI 预训练模型包含在本repo中.


## 安装
### 要求
- python == 3.6
- pytorch >= 1.1 (tested in 1.3.1)
- torchvision >= 0.3.0 (tested in 0.4.2)
- cuda (tested in cuda10)

```
conda create --name py36-sp python=3.6
conda activate py36-sp
pip install -r requirements.txt
pip install -r requirements_torch.txt # install pytorch
```

### 路径设置
- 数据集的路径 ($DATA_DIR), logs在 `setting.py` 中设置

### 数据集
数据集应下载到 $DATA_DIR 目录中. 合成shapes数据集也将在那里生成。文件夹结构应如下所示：

```
datasets/ ($DATA_DIR)
|-- COCO
|   |-- train2014
|   |   |-- file1.jpg
|   |   `-- ...
|   `-- val2014
|       |-- file1.jpg
|       `-- ...
`-- HPatches
|   |-- i_ajuntament
|   `-- ...
`-- synthetic_shapes  # 将自动生成
`-- KITTI (raw data的accumulated文件夹)
|   |-- 2011_09_26_drive_0020_sync
|   |   |-- image_00/
|   |   `-- ...
|   |-- ...
|   `-- 2011_09_28_drive_0001_sync
|   |   |-- image_00/
|   |   `-- ...
|   |-- ...
|   `-- 2011_09_29_drive_0004_sync
|   |   |-- image_00/
|   |   `-- ...
|   |-- ...
|   `-- 2011_09_30_drive_0016_sync
|   |   |-- image_00/
|   |   `-- ...
|   |-- ...
|   `-- 2011_10_03_drive_0027_sync
|   |   |-- image_00/
|   |   `-- ...
```
- MS-COCO 2014 
    - [MS-COCO 2014 link](http://cocodataset.org/#download)
- HPatches
    - [HPatches link](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)
- KITTI Odometry
    - [KITTI website](http://www.cvlibs.net/datasets/kitti/raw_data.php)
    - [download link](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip)



## 运行代码
- Notes:
    - 从任何步骤 (1-4) 开始，下载一些中间结果
    - 在一台 'NVIDIA 2080Ti'上进行训练通常需要8-10小时
    - 目前支持 'COCO' 数据集 (原始论文), 'KITTI' 数据集的训练
- Tensorboard:
    - log文件保存在 'runs/<\export_task>/...'下
    
`tensorboard --logdir=./runs/ [--host | static_ip_address] [--port | 6008]`

### 1) 关于合成shape的MagicPoint训练
```
python train4.py train_base configs/magicpoint_shapes_pair.yaml magicpoint_synth --eval
```
你不需要下载合成数据，您将在第一次运行它时生成它.
合成数据以 `./datasets` 的形式导出. 您可以在 `settings.py` 中更改设置.

### 2) 在 MS-COCO / kitti 上导出检测
这是 homography adaptation(HA) 的步骤，用于输出ground truth以进行联合训练.
- 确保配置文件中的预训练模型正确
- 确保COCO数据集唯一 '$DATA_DIR' (在setting.py中定义)
<!-- - 您可以通过编辑配置文件中的'task'来导出 hpatches 或 coco 数据集  -->
- 配置文件:
```
export_folder: <'train' | 'val'>  # 为训练或验证/评估导出
```
#### 全部的命令:
```
python export.py <export task>  <config file>  <export folder> [--outputImg | output images for visualization (space inefficient)]
```
#### 导出coco - 在训练集上
```
python export.py export_detector_homoAdapt configs/magicpoint_coco_export.yaml magicpoint_synth_homoAdapt_coco
```
#### 导出coco - 在验证集上
- Edit 'export_folder' to 'val' in 'magicpoint_coco_export.yaml'
```
python export.py export_detector_homoAdapt configs/magicpoint_coco_export.yaml magicpoint_synth_homoAdapt_coco
```
#### 导出kitti
- config配置
  - 检查配置文件中的 'root'  
  - train/ val split_files 包含在 `datasets/kitti_split/` 中
```
python export.py export_detector_homoAdapt configs/magicpoint_kitti_export.yaml magicpoint_base_homoAdapt_kitti
```
<!-- #### 导出tum
- config配置
  - 检查配置文件中的 'root' 
  - 将 'datasets/tum_split/train.txt' 设置为您拥有的序列
```
python export.py export_detector_homoAdapt configs/magicpoint_tum_export.yaml magicpoint_base_homoAdapt_tum
``` -->


### 3) 在 MS-COCO/ KITTI 上训练 superpoint
你需要fake ground truth标签来训练检测器detectors，标签可以从步骤 2) 导出，也可以从 [link](https://drive.google.com/drive/folders/1nnn0UbNMFF45nov90PJNnubDyinm2f26?usp=sharing) 下载. 然后，像往常一样，您需要在训练之前设置配置文件.
- config file配置文件
  - root: 指定您的root标签
  - root_split_txt: 放置 train.txt/ val.txt 分割文件的位置 (COCO不需要, KITTI需要)
  - labels: 从 homography adaptation 导出的标签
  - pretrained: 指定预训练模型 (可以从头开始训练)
- 'eval': 在训练期间打开验证/评估

#### 全部的命令
```
python train4.py <train task> <config file> <export folder> --eval
```

#### COCO
```
python train4.py train_joint configs/superpoint_coco_train_heatmap.yaml superpoint_coco --eval --debug
```
#### kitti
```
python train4.py train_joint configs/superpoint_kitti_train_heatmap.yaml superpoint_kitti --eval --debug
```

- 设置batch_size (最初为1)
- 请参阅: 'train_tutorial.md'

### 4) 导出/ 评估 HPatches 上的指标
- 使用预训练模型或在配置文件中指定模型
- ```./run_export.sh``` 将运行导出，然后进行验证/评估.

#### 导出
- 下载 HPatches 数据集 (上面的链接). 输入 $DATA_DIR .
```python export.py <export task> <config file> <export folder>```
- 导出 keypoints, descriptors, matching
```
python export.py export_descriptor  configs/magicpoint_repeatability_heatmap.yaml superpoint_hpatches_test
```
#### 验证/评估
```python evaluation.py <path to npz files> [-r, --repeatibility | -o, --outputImg | -homo, --homography ]```
- 评估单应性估计 homography estimation/ 重复性 repeatability/ 匹配分数 matching scores ...
```
python evaluation.py logs/superpoint_hpatches_test/predictions --repeatibility --outputImg --homography --plotMatching
```

### 5) 导出/ Evaluate repeatability on SIFT
- 请参阅另一个项目 : [Feature-preserving image denoising with multiresolution filters](https://github.com/eric-yyjau/image_denoising_matching)
```shell
# 导出 detection, description, matching
python export_classical.py export_descriptor configs/classical_descriptors.yaml sift_test --correspondence

# 评估 (use 'sift' flag)
python evaluation.py logs/sift_test/predictions --sift --repeatibility --homography 
```


- 指定预训练模型

## 预训练模型
### 当前最佳模型
- *COCO 数据集*
```logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar```
- *KITTI 数据集*
```logs/superpoint_kitti_heat2_0/checkpoints/superPointNet_50000_checkpoint.pth.tar```
### 来自 magicleap 的模型
```pretrained/superpoint_v1.pth```

## Jupyter notebook 
```shell
# 显示保存在文件夹中的图像
jupyter notebook
notebooks/visualize_hpatches.ipynb 
```

## 更新 (年.月.日)
- 2020.08.05: 
  - 从 (https://github.com/eric-yyjau/pytorch-superpoint/pull/19) 更新pytorch nms
  - 在google drive上更新和测试KITTI数据加载器和标签 (应该能够适应 KITTI 原始格式)
  - 在第5步更新和测试SIFT验证/评估.

## 已知问题
- ~~test step 5: evaluate on SIFT测试步骤5： 在SIFT上评估~~
- 以低分辨率(240x320)而不是高分辨率(480x640)导出 COCO 数据集
- 由于步骤 1 早就完成了，我们仍在进行第2-4步的测试。请参考我们的预训练和导出标签，或者让我们知道整个pipeline时如何工作的。
- 来自tensorboard的警告.

## 正在进行的工作
- 发布带有unit测试的 notebooks
- 数据集: ApolloScape/ TUM.

## 引用
请引用原文.
```
@inproceedings{detone2018superpoint,
  title={Superpoint: Self-supervised interest point detection and description},
  author={DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={224--236},
  year={2018}
}
```

也请引用我们的文章 DeepFEPE .
```
@misc{2020_jau_zhu_deepFEPE,
Author = {You-Yi Jau and Rui Zhu and Hao Su and Manmohan Chandraker},
Title = {Deep Keypoint-Based Camera Pose Estimation with Geometric Constraints},
Year = {2020},
Eprint = {arXiv:2007.15122},
}
```

# Credits
此实现是由 [You-Yi Jau](https://github.com/eric-yyjau) 和 [Rui Zhu](https://github.com/Jerrypiglet) 开发的. 如有任何问题，请联系You Yi. 
同样，这项工作是基于 [Rémi Pautrat](https://github.com/rpautrat) 和 [Paul-Edouard Sarlin](https://github.com/Skydes) 以及 [SuperPointPretrainedNetwork](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork) 实现的
感谢Daniel DeTone在实现过程中提供的帮助

## Posts
[What have I learned from the implementation of deep learning paper?](https://medium.com/@eric.yyjau/what-have-i-learned-from-the-implementation-of-deep-learning-paper-365ee3253a89)
