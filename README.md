# Pytorch-Deeplabv2-cityscapes
---------
Deeplab_V2 (./train_cityscapes.py)
    1. CrossEntropy Loss ~ 0.3
    2. IOU ~ 64 on validation (unstable due to random crop in validation dataset)
    3. CRF is NOT included

### Requirements
```
- python==3.5+
- pytorch==0.4.0
- tensorboardx
- docstr
- opencv
- PIL
```
## Usage
---------

### Dataset
make sure the dataset file are in the following structure, using cityscapes as example
```
-cityscapes
	- leftImg8bit
	 	- train
			- *.png
		- test
			- *.png
		- val
			- *.png
	- gtFine
		- train
			- *.png
		- test
			- *.png
		- val
			- *.png
```

To train the model, change `dataset_path` in the `create_namelist_cityscapes.sh` and point to a dir in the format shown above.
optional
change the `docstr` in the `train_cityscapes` 

```
python train_cityscapes.py
``` 

to validate the result, comment `model.train()` in the main funtion and run it. 

### TODO
- reduce hard code input
- change the input logic
- add multiple scale auxilary loss
- Enable mutiple GPUs calculation 
- add reference 
- add progress bar



