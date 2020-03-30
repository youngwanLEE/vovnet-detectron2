# [VoVNet-v2](https://github.com/youngwanLEE/CenterMask) backbone networks in [Detectron2](https://github.com/facebookresearch/detectron2)
**Efficient Backbone Network for Object Detection and Segmentation**



[[`CenterMask(code)`](https://github.com/youngwanLEE/CenterMask)][[`CenterMask2(code)`](https://github.com/youngwanLEE/centermask2)] [[`VoVNet-v1(arxiv)`](https://arxiv.org/abs/1904.09730)] [[`VoVNet-v2(arxiv)`](https://arxiv.org/abs/1911.06667)] [[`BibTeX`](#CitingVoVNet)]


<div align="center">
  <img src="https://dl.dropbox.com/s/jgi3c5828dzcupf/osa_updated.jpg" width="700px" />
</div>

  
  
In this project, we release code for **VoVNet-v2** backbone network (introduced by [CenterMask](https://arxiv.org/abs/1903.12174)) in [detectron2](https://github.com/facebookresearch/detectron2) as a [extention form](https://github.com/youngwanLEE/detectron2/tree/vovnet/projects/VoVNet).
VoVNet can  extract diverse feature representation *efficiently* by using One-Shot Aggregation (OSA) module that concatenates subsequent layers at once. Since the OSA module can capture multi-scale receptive fields, the diversifed feature maps allow object detection and segmentation to address multi-scale objects and pixels well, especially robust on small objects. VoVNet-v2 improves VoVNet-v1 by adding identity mapping that eases the optimization problem and *effective* SE (Squeeze-and-Excitation) that enhances the diversified feature representation.

## Highlight
Compared to ResNe(X)t backbone
- ***Efficient*** : Faster speed
- ***Accurate*** : Better performance, especially *small* object.

## Update
- *Lightweight*-VoVNet-19 has been released. (19/02/2020)
- VoVNetV2-19-FPNLite has been released. (22/01/2020)
- [centermask2](https://github.com/youngwanLEE/centermask2) has been released. (20/02/2020)
## Results on MS-COCO in Detectron2

### Note

We measure the inference time of all models with batch size 1 on the same V100 GPU machine.  
We train all models using V100 8GPUs.

- pytorch1.3.1
- CUDA 10.1
- cuDNN 7.3

### Faster R-CNN

#### Lightweight-VoVNet with _FPNLite_

|Backbone|Param.|lr sched|inference time|AP|APs|APm|APl|download|
|:--------:|:---:|:---:|:--:|--|----|----|---|--------|
|MobileNetV2|3.5M|3x|0.022|33.0|19.0|35.0|43.4|<a href="https://dl.dropbox.com/s/q4iceofvlcu207c/faster_mobilenetv2_FPNLite_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/tz60e7rtnbsrdgd/faster_mobilenetv2_FPNLite_ms_3x_metrics.json">metrics</a>
||
|V2-19|11.2M|3x|0.034|38.9|24.8|41.7|49.3|<a href="https://www.dropbox.com/s/u5pvmhc871ohvgw/fast_V_19_eSE_FPNLite_ms_3x.pth?dl=1">model</a>&nbsp;\|&nbsp;<a href="https://www.dropbox.com/s/riu7hkgzlmnndhc/fast_V_19_eSE_FPNLite_ms_3x_metrics.json">metrics</a>
|V2-19-**DW**|6.5M|3x|0.027|36.7|22.7|40.0|46.0|<a href="https://www.dropbox.com/s/7h6zn0owumucs48/faster_rcnn_V_19_eSE_dw_FPNLite_ms_3x.pth?dl=1">model</a>&nbsp;\|&nbsp;<a href="https://www.dropbox.com/s/627hf4h1m485926/faster_rcnn_V_19_eSE_dw_FPNLite_ms_3x_metrics.json">metrics</a>
|V2-19-**Slim**|3.1M|3x|0.023|35.2|21.7|37.3|44.4|<a href="https://www.dropbox.com/s/yao1i32zdylx279/faster_rcnn_V_19_eSE_slim_FPNLite_ms_3x.pth?dl=1">model</a>&nbsp;\|&nbsp;<a href="https://www.dropbox.com/s/jrgxltneki9hk84/faster_rcnn_V_19_eSE_slim_FPNLite_ms_3x_metrics.json">metrics</a>
|V2-19-**Slim**-**DW**|1.8M|3x|0.022|32.4|19.1|34.6|41.8|<a href="https://www.dropbox.com/s/blpjx3iavrzkygt/faster_rcnn_V_19_eSE_slim_dw_FPNLite_ms_3x.pth?dl=1">model</a>&nbsp;\|&nbsp;<a href="https://www.dropbox.com/s/3og68zhq2ubr7mu/faster_rcnn_V_19_eSE_slim_dw_FPNLite_ms_3x_metrics.json">metrics</a>

* _**DW** and **Slim** denote depthwise separable convolution and a thiner model with half the channel size, respectively._                              


|Backbone|Param.|lr sched|inference time|AP|APs|APm|APl|download|
|:--------:|:---:|:---:|:--:|--|----|----|---|--------|
|V2-19-FPN|37.6M|3x|0.040|38.9|24.9|41.5|48.8|<a href="https://www.dropbox.com/s/1rfvi6vzx45z6y5/faster_V_19_eSE_ms_3x.pth?dl=1">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/dq7406vo22wjxgi/faster_V_19_eSE_ms_3x_metrics.json">metrics</a>
||
|R-50-FPN|51.2M|3x|0.047|40.2|24.2|43.5|52.0|<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/metrics.json">metrics</a>
|**V2-39-FPN**|52.6M|3x|0.047|42.7|27.1|45.6|54.0|<a href="https://dl.dropbox.com/s/dkto39ececze6l4/faster_V_39_eSE_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/dx9qz1dn65ccrwd/faster_V_39_eSE_ms_3x_metrics.json">metrics</a>
||
|R-101-FPN|70.1M|3x|0.063|42.0|25.2|45.6|54.6|<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/metrics.json">metrics</a>
|**V2-57-FPN**|68.9M|3x|0.054|43.3|27.5|46.7|55.3|<a href="https://dl.dropbox.com/s/c7mb1mq10eo4pzk/faster_V_57_eSE_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/3tsn218zzmuhyo8/faster_V_57_eSE_metrics.json">metrics</a>
||
|X-101-FPN|114.3M|3x|0.120|43.0|27.2|46.1|54.9|<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/metrics.json">metrics</a>|
|**V2-99-FPN**|96.9M|3x|0.073|44.1|28.1|47.0|56.4|<a href="https://dl.dropbox.com/s/v64mknwzfpmfcdh/faster_V_99_eSE_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/zvaz9s8gvq2mhrd/faster_V_99_eSE_ms_3x_metrics.json">metrics</a>|

### Mask R-CNN

|Backbone|lr sched|inference time|box AP|box APs|box APm|box APl|mask AP|mask APs|mask APm|mask APl|download|
|:--------:|:--------:|:--:|--|----|----|---|--|----|----|---|--------|
|V2-19-**FPNLite**|3x|0.036|39.7|25.1|42.6|50.8|36.4|19.9|38.8|50.8|<a href="https://www.dropbox.com/s/h1khv9l7quakvz0/mask_V_19_eSE_FPNLite_ms_3x.pth?dl=1">model</a>&nbsp;\|&nbsp;<a href="https://www.dropbox.com/s/8fophrb1f1mf9ih/mask_V_19_eSE_FPNLite_ms_3x_metrics.json">metrics</a>
|V2-19-FPN|3x|0.044|40.1|25.4|43.0|51.0|36.6|19.7|38.7|51.2|<a href="https://www.dropbox.com/s/dyeyuag5va96tqo/mask_V_19_eSE_ms_3x.pth?dl=1">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/0y0q97gi8u8kq2n/mask_V_19_eSE_ms_3x_metrics.json">metrics</a>
||
|R-50-FPN|3x|0.055|41.0|24.9|43.9|53.3|37.2|18.6|39.5|53.3|<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/metrics.json">metrics</a>
|**V2-39-FPN**|3x|0.052|43.8|27.6|47.2|55.3|39.3|21.4|41.8|54.6|<a href="https://dl.dropbox.com/s/c5o3yr6lwrb1170/mask_V_39_eSE_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/21xqlv1ofn7oa1z/mask_V_39_eSE_metrics.json">metrics</a>
||
|R-101-FPN|3x|0.070|42.9|26.4|46.6|56.1|38.6|19.5|41.3|55.3|<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/metrics.json">metrics</a>
|**V2-57-FPN**|3x|0.058|44.2|28.2|47.2|56.8|39.7|21.6|42.2|55.6|<a href="https://dl.dropbox.com/s/aturknfroupyw92/mask_V_57_eSE_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/8sdek6hkepcu7na/mask_V_57_eSE_metrics.json">metrics</a>
||
|X-101-FPN|3x|0.129|44.3|27.5|47.6|56.7|39.5|20.7|42.0|56.5|<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/metrics.json">metrics</a>|
|**V2-99-FPN**|3x|0.076|44.9|28.5|48.1|57.7|40.3|21.7|42.8|56.6|<a href="https://dl.dropbox.com/s/qx45cnv718k4zmn/mask_V_99_eSE_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/u1sav8deha47odp/mask_V_99_eSE_metrics.json">metrics</a>|


### Panoptic-FPN on COCO
<!--
./gen_html_table.py --config 'COCO-PanopticSegmentation/*50*' 'COCO-PanopticSegmentation/*101*'  --name R50-FPN R50-FPN R101-FPN --fields lr_sched train_speed inference_speed mem box_AP mask_AP PQ
-->


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">PQ</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: panoptic_fpn_R_50_3x -->
<tr><td align="left">R-50-FPN</td>
<td align="center">3x</td>
<td align="center">0.063</td>
<td align="center">40.0</td>
<td align="center">36.5</td>
<td align="center">41.5</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/metrics.json">metrics</a></td>
</tr>
<!-- ROW: panoptic_fpn_V_39_3x -->
<tr><td align="left">V2-39-FPN</td>
<td align="center">3x</td>
<td align="center">0.063</td>
<td align="center">42.8</td>
<td align="center">38.5</td>
<td align="center">43.4</td>
<td align="center"><a href="https://www.dropbox.com/s/fnr9r4arv0cbfbf/panoptic_V_39_eSE_3x.pth?dl=1">model</a>&nbsp;|&nbsp;<a href="https://dl.dropbox.com/s/vftfukrjuu7w1ao/panoptic_V_39_eSE_3x_metrics.json">metrics</a></td>
</tr>
<!-- ROW: panoptic_fpn_R_101_3x -->
<tr><td align="left">R-101-FPN</td>
<td align="center">3x</td>
<td align="center">0.078</td>
<td align="center">42.4</td>
<td align="center">38.5</td>
<td align="center">43.0</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/metrics.json">metrics</a></td>
</tr>
<!-- ROW: panoptic_fpn_V_57_3x -->
 <tr><td align="left">V2-57-FPN</td>
<td align="center">3x</td>
<td align="center">0.070</td>
<td align="center">43.4</td>
<td align="center">39.2</td>
<td align="center">44.3</td>
<td align="center"><a href="https://www.dropbox.com/s/zhoqx5rvc0jj0oa/panoptic_V_57_eSE_3x.pth?dl=1">model</a>&nbsp;|&nbsp;<a href="https://dl.dropbox.com/s/20hwrmru15dilre/panoptic_V_57_eSE_3x_metrics.json">metrics</a></td>
</tr>
</tbody></table>


Using this command with `--num-gpus 1`
```bash
python /path/to/vovnet-detectron2/train_net.py --config-file /path/to/vovnet-detectron2/configs/<config.yaml> --eval-only --num-gpus 1 MODEL.WEIGHTS <model.pth>
```

## Installation

As this vovnet-detectron2 is implemented as a [extension form](https://github.com/youngwanLEE/detectron2/tree/vovnet/projects/VoVNet) (detectron2/projects) upon detectron2, you just install [detectron2](https://github.com/facebookresearch/detectron2) following [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

Prepare for coco dataset following [this instruction](https://github.com/facebookresearch/detectron2/tree/master/datasets).

## Training

#### ImageNet Pretrained Models

We provide backbone weights pretrained on ImageNet-1k dataset.
* [VoVNetV2-19-Slim-DW](https://www.dropbox.com/s/f3s7ospitqoals1/vovnet19_ese_slim_dw_detectron2.pth)
* [VoVNetV2-19-Slim](https://www.dropbox.com/s/8h5ybmi4ftbcom0/vovnet19_ese_slim_detectron2.pth)
* [VoVNetV2-19-DW](https://www.dropbox.com/s/9awvl0mxye3nqz1/vovnet19_ese_dw_detectron2.pth)
* [VoVNetV2-19](https://dl.dropbox.com/s/rptgw6stppbiw1u/vovnet19_ese_detectron2.pth)
* [VoVNetV2-39](https://dl.dropbox.com/s/q98pypf96rhtd8y/vovnet39_ese_detectron2.pth)
* [VoVNetV2-57](https://dl.dropbox.com/s/8xl0cb3jj51f45a/vovnet57_ese_detectron2.pth)
* [VoVNetV2-99](https://dl.dropbox.com/s/1mlv31coewx8trd/vovnet99_ese_detectron2.pth)


To train a model, run
```bash
python /path/to/vovnet-detectron2/train_net.py --config-file /path/to/vovnet-detectron2/configs/<config.yaml>
```

For example, to launch end-to-end Faster R-CNN training with VoVNetV2-39 backbone on 8 GPUs,
one should execute:
```bash
python /path/to/vovnet-detectron2/train_net.py --config-file /path/to/vovnet-detectron2/configs/faster_rcnn_V_39_FPN_3x.yaml --num-gpus 8
```

## Evaluation

Model evaluation can be done similarly:
```bash
python /path/to/vovnet-detectron2/train_net.py --config-file /path/to/vovnet-detectron2/configs/faster_rcnn_V_39_FPN_3x.yaml --eval-only MODEL.WEIGHTS <model.pth>
```

## TODO
 - [x] Adding Lightweight models
 - [ ] Applying VoVNet for other meta-architectures



## <a name="CitingVoVNet"></a>Citing VoVNet

If you use VoVNet, please use the following BibTeX entry.

```BibTeX
@inproceedings{lee2019energy,
  title = {An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection},
  author = {Lee, Youngwan and Hwang, Joong-won and Lee, Sangrok and Bae, Yuseok and Park, Jongyoul},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year = {2019}
}

@article{lee2019centermask,
  title={CenterMask: Real-Time Anchor-Free Instance Segmentation},
  author={Lee, Youngwan and Park, Jongyoul},
  booktitle={CVPR},
  year={2020}
}
```
