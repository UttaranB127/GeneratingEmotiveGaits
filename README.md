# Generating Emotive Gaits for Virtual Agents Using Affect-Based Autoregression

This is the official implementation of the paper [Generating Emotive Gaits for Virtual Agents Using Affect-Based Autoregression](https://ieeexplore.ieee.org/abstract/document/9284667?casa_token=6fxtgPaWUL0AAAAA:NGaxEPu34BbmXRwuOoZe7WQV5zv1wKct0ZXqBg-y8P_Y7DTn76a3EDHwLnFlkZuR9cCHBJU9vBA). Please add the following citation if you find our work uesful:

```
@inproceedings{bhattacharya2020generating,
author = {Uttaran Bhattacharya and Nicholas Rewkowski and Pooja Guhan and Niall L. Williams and Trisha Mittal and Aniket Bera and Dinesh Manocha},
title = {Generating Emotive Gaits for Virtual Agents Using Affect-Based Autoregression},
booktitle = {2020 {IEEE} International Symposium on Mixed and Augmented Reality, {ISMAR} 2020, November 9-13, 2020},
publisher = {{IEEE}},
year      = {2020}
}
```

Our code is tested on Ubuntu 18.04 LTS with python 3.7.

## Installation

1. Unzip this repository.

We use $BASE to refer to the base directory for this project (the directory containing `main.py`). Change present working directory to $BASE.

2. [Optional but recommended] Create a conda envrionment for the project and activate it.

```
conda create gen-emo-gaits-env python=3.7
conda activate gen-emo-gaits-env
```

3. Install the package requirements.

```
pip install -r requirements.txt
```
Note: You might need to manually uninstall and reinstall `matplotlib` and `kiwisolver` for them to work.

4. Install PyTorch following the [official instructions](https://pytorch.org/).
Note: You might need to manually uninstall and reinstall `numpy` for `torch` to work.

## Downloading the dataset
We downloaded the [Edinburgh Locmotion MOCAP Database](https://bitbucket.org/jonathan-schwarz/edinburgh_locomotion_mocap_dataset/src/master/), and annotated the locomotion data with emotion labels. Our processed and annotated dataset is available for download here: https://drive.google.com/file/d/1-D_tT09a7CfO3JSSHWX7-Ao3SjCxTsD2/view?usp=sharing. Unzip the downloaded file to a directorty named "data", located at the same level $BASE (i.e., $BASE and the data are sibling directories).

## Running the code
Run the `main.py` file with the appropriate command line arguments.
```
python main.py <args list>
```

The full list of arguments is available inside `main.py`.

For any argument not specificed in the command line, the code uses the default value for that argument.

On running `main.py`, the code will train the network and generate sample gestures post-training.

We also provide a pretrained model inside the `models` directory, available for download here: https://drive.google.com/file/d/1u8Ed_iPXdc-8bVzyRqXOfQ5g2QogNOj1/view?usp=sharing. Save the `models` directory directory under $BASE.

Set the command-line argument `--train` to `False` to skip training and use this model directly for evaluation. The generated samples are stored in the automatically created `render` directory. We generate test samples by deafult and also store the corresponding ground truth samples for comparison. We have tested that the samples, stored in `.bvh` files, are compatible with blender.
