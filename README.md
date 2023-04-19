# MLIA-2022-final
MLIA final project, fall 2022

Shantanu Sanyal, Allan Wan, Isabelle Liu, Julian Nguyen

SWIN UNETR for semantic segmentation of heart MRI implementation

https://arxiv.org/pdf/2201.01266.pdf

Installation:

python -m pip -e [path to MLIA-2022-FINAL directory]

Usage:

Augmentation:
The multiplier argument determines how many times each input image will be augmented.  For example, a multiplier of 2 on the original dataset will produce 112*2 = 224 augmented images.

To run augmentation, run the augmentation python script:

$ python3 <path to MLIA-2022-FINAL>/scripts/generate_augmented_data.py --multiplier 2 --input-mask <path to CardiacImage_data/Training/train_myocardium_segmentation> --input-data <path to CardiacImageData/Training/train_imageData> --output-mask <desired output folder for augmented segmentations> --output-data <desired output folder for augmented input images>

To run augmentation in Rivanna, launch the augmentation slurm script like so:

$ sbatch <path to MLIA-2022-FINAL>/scripts/augmentation.slurm <path to MLIA-2022-Final> <path to CardiacImage_data/Training> <multiplier>


Training:

** Note that the configs/basic.json config file has defaults loaded that should work with an unmodified unizpped CardiacImage_data folder.

To run training, you will need to set some parameters to ensure that the correct data is loaded.  Edit or copy the config file under MLIA-2022-FINAL/configs/basic.json and ensure the following:

Under "swin", "histogram_matching_reference" should point to the filepath of the image you want to histogram match.  We found image 'im107.png' under 'CardiacImage_data/Training/train_imageData' to be appropriate.  Note that putting "default" will load a stored copy of im107.png, while leaving the field as an empty string will cause the network to skip histogram matching.

Under "hyperparameters", the X_data_folder and Y_data_folder need to be set to the folder names for the X data (input images) and Y data (mask images).  The default folder names are "train_imageData" and "train_myocardium_segementations"

One can then launch training with 
$ python3 <path to MLIA-2022-FINAL>/src/swin/MLIA-main.py -train --net-cfg <path to json config file> --input <path to CardiacImage_data/Training> --output /scratch/$USER/network_out

##RIVANNA TRAINING

Set up:
After getting the repo into your /scratch/$USER Rivanna location, extract the T4 data into a seperate location under /scratch/$USER.

Prepare the config file as described above

Finally, training can be launched with 
$ sbatch <path to MLIA-2022-FINAL>/scripts/run_training.slurm <path to MLIA-2022-FINAL> <path to config file> <path to output directory>



Inference:
To run inference on a dataset, a config file similar to the one for trainig is needed.  A default config file is provided under configs/inference.json

To run inference,

$ python <path to MLIA-2022-FINAL>/src/swin/MLIA-main.py --inference --net-cfg <path to json config> --input <path to CardiacImage_data/Testing1-withlabel> --output <path to output directory>
  
No Rivanna implementation is needed for inference.
