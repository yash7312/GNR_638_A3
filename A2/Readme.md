This is the directory for Assignment 2 

### Before you run the code

Download the floders containing saved models from the following links:

checkpoints : https://drive.google.com/file/d/1IYl6swM6_GbS3NcDV8bVbJuZa5aKBia8/view?usp=sharing

checkpoints_ft : https://drive.google.com/file/d/1LIp5fCY6HY_tDCaSNwcuJFtm_DsAzACW/view?usp=sharing

After downloading the zip files and extract them . Place them in the root directory of this assignment (i.e. the same directory as this README file). The directory structure should become like :

checkpoints

├── efficientnet_b0_linear_probe.pth

├── inception_v3_linear_probe.pth

└── resnet50_linear_probe.pth

checkpoints_ft

├── efficientnet_b0_FullFT.pth

├── efficientnet_b0_LastBlockFT.pth

├── efficientnet_b0_LinearProbe.pth

├── efficientnet_b0_Selective20.pth

├── inception_v3_FullFT.pth

├── inception_v3_LastBlockFT.pth

├── inception_v3_LinearProbe.pth

├── inception_v3_Selective20.pth

├── resnet50_FullFT.pth

├── resnet50_LastBlockFT.pth

├── resnet50_LinearProbe.pth

└── resnet50_Selective20.pth

Other files in the directory remains the same .

### Running the code

Part-1 :   ```python linear_probe.py```

Part-2 :   ```python fine_tuning.py```

Part-3 :   ```python 4_3_few_shot.py```

Part-4 :   ```python corruption_robust_eval.py```
 
Part-5 :   ```python Layerwise_probing.py```

### Testing for Hidden Test cases

To test for hidden cases please name your folder train_data in that have a folder train_data and in that have subfolders for each class. Since We am reading data from './train_data/train_data' . 
The structure should look like:

train_data

└── train_data
    
    └── Subfolders for each class
