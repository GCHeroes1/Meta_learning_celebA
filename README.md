Download the CelebA folder from here:

https://drive.google.com/drive/folders/1ngzl-cI1s-Ib8k_e9LDVZ_eQJRHvXq2V

It may be downloaded into separate files, but unzipping the one indexed by 001 will create the CelebA folder
containing the Anno and Img folders. The Img folder contains the img_align_celeb.zip folder which also needs to be
unzipped. Once completed update the dataroot and labels_path variables in celeb_dataset_creation.py

Download the mini-imagenet dataset from here and add them to a ./data/mini-imagenet folder

test:
https://www.dropbox.com/s/ye9jeb5tyz0x01b/mini-imagenet-cache-test.pkl?dl=1

train:
https://www.dropbox.com/s/9g8c6w345s2ek03/mini-imagenet-cache-train.pkl?dl=1

validation:
https://www.dropbox.com/s/ip1b7se3gij3r1b/mini-imagenet-cache-validation.pkl?dl=1

Download tiered-imagenet from here and add it to ./data/tiered-imagenet/ and unzip the tar file

https://drive.google.com/u/0/uc?id=1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07&export=download

The final /data folder structure should look as follows:

```angular2html
.
├── ...
├── data
│   ├── CelebA-*-001
│   │   └── CelebA
│   │       ├── Anno
│   │       │   └── identity_celebA.txt
│   │       └── Img
│   │           └── img_align_celeba
│   │               └── img_align_celeba
│   │                   └── ...
│   ├── fc100
│   ├── mini-imagenet
│   │   ├── mini-imagenet-cache-test.pkl
│   │   ├── mini-imagenet-cache-train.pkl
│   │   └── mini-imagenet-cache-validation.pkl
│   ├── omniglot
│   └── tiered-imagenet
│       └── tiered-imagenet.tar
└── ...

```

The code is written to run in python 3.9 with pytorch for cuda 11.3. To setup the virtualenv, follow the steps below:

```
python -m venv ./venv
python ./venv/Scripts/activate
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -r requirements.txt
```

To generate example images, run `python celebA_dataset_creation.py`

To run the experiments described in the study, run `python experiments.py`

To plot the results, run `python plot_from_files.py`
