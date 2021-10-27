# paddle-superpoint

这是论文 "SuperPoint: Self-Supervised Interest Point Detection and Description." [ArXiv 2018](https://arxiv.org/abs/1712.07629) 的Paddle实现. 

此代码部分基于tensorflow实现
https://github.com/rpautrat/SuperPoint.

此repo是此paper [deepFEPE(IROS 2020)](https://github.com/eric-yyjau/pytorch-deepFEPE.git) 的bi-product .

## 我们的实现与原始论文之间的差异
- *Descriptor loss*: 我们使用不同的方法描述loss，包括密集方法（如paper所示，但略有不同）和稀疏方法。稀疏损失可以更有效地收敛到类似地性能，这里的默认设置是稀疏方法.

## HPatches上的结果
| 任务                                       | Homography estimation |      |      | Detector metric |      | Descriptor metric |                |
|-------------------------------------------|-----------------------|------|------|-----------------|------|-------------------|----------------|
|                                           | Epsilon = 1           | 3    | 5    | Repeatability   | MLE  | NN mAP            | Matching Score |
| 预训练模型                                  | 0.44                  | 0.77 | 0.83 | 0.606           | 1.14 | 0.81              | 0.55           |
| superpoint_coco_heat2_0_170k_hpatches_sub | 0.46                  | 0.75 | 0.81 | 0.63            | 1.07 | 0.78              | 0.42           |


- 预训练模型来自 [SuperPointPretrainedNetwork](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork).
- 验证是在我们的验证脚本下完成的
- COCO 训练模型包含在本repo中.


## 安装
### 要求
- python == 3.8
- paddle >= 2.1 (tested in 2.1.2)
- cuda (tested in cuda10)


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
```
- MS-COCO 2014 
    - [MS-COCO 2014 link](http://cocodataset.org/#download)
- HPatches
    - [HPatches link](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)



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
不需要下载合成数据，将在第一次运行它时生成它.
合成数据以 `./datasets` 的形式导出. 可以在 `settings.py` 中更改设置.

### 2) 在 MS-COCO 上导出检测
这是 homography adaptation(HA) 的步骤，用于输出ground truth以进行训练.
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


### 3) 在 MS-COCO 上训练 superpoint
需要fake ground truth标签来训练detectors，标签可以从步骤 2) 导出，也可以从 [link](https://drive.google.com/drive/folders/1nnn0UbNMFF45nov90PJNnubDyinm2f26?usp=sharing) 下载. 然后，像往常一样，您需要在训练之前设置配置文件.
- config file配置文件
  - root: 指定root标签
  - root_split_txt: 放置 train.txt/ val.txt 分割文件的位置
  - labels: 从 homography adaptation 导出的标签
  - pretrained: 指定预训练模型 (可以从头开始训练)
- 'eval': 在训练期间打开验证

#### 全部的命令
```
python train4.py <train task> <config file> <export folder> --eval
```

#### COCO
```
python train4.py train_joint configs/superpoint_coco_train_heatmap.yaml superpoint_coco --eval --debug
```

- 设置batch_size (最初为1)
- 请参阅: 'train_tutorial.md'

### 4) 导出/ 验证 HPatches 上的指标
- 使用预训练模型或在配置文件中指定模型
- ```./run_export.sh``` 将运行导出，然后进行验证/评估.

#### 导出
- 下载 HPatches 数据集 (上面的链接). 输入 $DATA_DIR .
```python export.py <export task> <config file> <export folder>```
- 导出 keypoints, descriptors, matching
```
python export.py export_descriptor  configs/magicpoint_repeatability_heatmap.yaml superpoint_hpatches_test
```
#### 验证
```python evaluation.py <path to npz files> [-r, --repeatibility | -o, --outputImg | -homo, --homography ]```
- 评估单应性估计 homography estimation/ 重复性 repeatability/ 匹配分数 matching scores ...
```
python evaluation.py logs/superpoint_hpatches_test/predictions --repeatibility --outputImg --homography --plotMatching
```


- 指定预训练模型

## 预训练模型
### 当前最佳模型
- *COCO 数据集*
```logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pdparams```
### 来自 magicleap 的模型
```pretrained/superpoint_v1.pth```