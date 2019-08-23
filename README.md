# A Simple demo for NVIDIA/flownet2-pytorch

## USAGE

### 1. Clone
```bash
# get flownet2-pytorch source code
git clone https://github.com/NVIDIA/flownet2-pytorch.git
cd flownet2-pytorch

# get demo source code
# and copy *.py to 'flownet2-pytorch' folder
git clone https://github.com/jjkislele/i_just_want_a_simple_demo_for_flownet2
cp ./i_just_want_a_simple_demo_for_flownet2/*.py .
```

### 2. Install custom layers
flownet2-pytorch supports python3. I recommend anaconda to manage these packages and virtual envoriments.

It works for me:

- gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.11)
- python 3.6.9
- pytorch 1.1.0
- cuda 10, cudatoolkit 10.0.130
- tensorboardx 1.8
- colorama 0.4.1
- tqdm 4.32.1
- setproctitle 1.1.10

```bash
conda install -c conda-forge tensorboardx 
conda install -c conda-forge setproctitle 
...

sh install.sh
```

### 3. Download pre-trained model

**Should you use these pre-trained weights, please adhere to the [license agreements](https://drive.google.com/file/d/1TVv0BnNFh3rpHZvD-easMb9jYrPE2Eqd/view?usp=sharing).**

**NEW** [BaiduYun](https://pan.baidu.com/s/12p4IE-xiNi6OHMsDpLBkug) [Fetch Code: ncm4]

* [FlowNet2](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing)[620MB]
* [FlowNet2-C](https://drive.google.com/file/d/1BFT6b7KgKJC8rA59RmOVAXRM_S7aSfKE/view?usp=sharing)[149MB]
* [FlowNet2-CS](https://drive.google.com/file/d/1iBJ1_o7PloaINpa8m7u_7TsLCX0Dt_jS/view?usp=sharing)[297MB]
* [FlowNet2-CSS](https://drive.google.com/file/d/157zuzVf4YMN6ABAQgZc8rRmR5cgWzSu8/view?usp=sharing)[445MB]
* [FlowNet2-CSS-ft-sd](https://drive.google.com/file/d/1R5xafCIzJCXc8ia4TGfC65irmTNiMg6u/view?usp=sharing)[445MB]
* [FlowNet2-S](https://drive.google.com/file/d/1V61dZjFomwlynwlYklJHC-TLfdFom3Lg/view?usp=sharing)[148MB]
* [FlowNet2-SD](https://drive.google.com/file/d/1QW03eyYG_vD-dT-Mx4wopYvtPu_msTKn/view?usp=sharing)[173MB]

### 4. Inference

```bash
python demo.py --input_dir dataset/video/ --resume checkpoints/FlowNet2_checkpoint.pth.tar --save result --save_flow --save_img
```

File structure

```
.
├── dataset
│   └── video
│         ├── 1.jpg
│         ├── 2.jpg
│         └── *.jpg
├── checkpoints		
│   └── FlowNet2_checkpoint.pth.tar
├── demo.py 
├── flowlib.py  	
└── README.md 	
```

For more help, please type 

```bash
python main.py --help
```

### 5. Result

If the demo runs properly, several folders will be created under your workspace:

```
.
├── result
│   └── video
│         ├── 1.jpg
│         ├── 2.jpg
│         └── *.jpg
├── checkpoints		
│   └── FlowNet2_checkpoint.pth.tar
├── demo.py 
├── flowlib.py  	
├── README.md
├── *result 
│   ├── 1.jpg.flo
│   ├── 2.jpg.flo
│   └── *.jpg.flo
└── *result_img 
    ├── 1.jpg
    ├── 2.jpg
    └── *.jpg
```

``./result/1.jpg.flo`` is the flow inference between ``1.jpg`` and ``2.jpg``.
![Predicted flows](./result.png)

## Acknowledgments

- [ClementPinard/FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch)
- [NVIDIA/flownet2-pytorch](https://https://github.com/NVIDIA/flownet2-pytorch)
- [vt-vl-lab/pytorch_flownet2](https://github.com/vt-vl-lab/pytorch_flownet2)
- [liruoteng/OpticalFlowToolkit](https://github.com/liruoteng/OpticalFlowToolkit)