

nohup sh shell/all_exp.sh &
<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->·


<!-- ABOUT THE PROJECT -->
## TS-ACL: A Time Series Analytic Continual Learning Framework for Privacy-Preserving and Class-Incremental Pattern Recognition 


## Requirements
![](https://img.shields.io/badge/python-3.10-green.svg)

![](https://img.shields.io/badge/pytorch-1.13.1-blue.svg)
![](https://img.shields.io/badge/ray-2.3.1-blue.svg)
![](https://img.shields.io/badge/PyYAML-6.0-blue.svg)
![](https://img.shields.io/badge/scikit--learn-1.0.2-blue.svg)
![](https://img.shields.io/badge/matplotlib-3.7.1-blue.svg)
![](https://img.shields.io/badge/pandas-1.5.3-blue.svg)
![](https://img.shields.io/badge/seaborn-0.12.2-blue.svg)

### Create Conda Environment

1. Create the environment from the file
   ```sh
   conda env create -f environment.yml
   ```

2. Activate the environment `tscl`
   ```sh
   conda activate tscl
   ```
----

## Dataset
### Available Datasets
1. [UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
2. [UWAVE](http://www.timeseriesclassification.com/description.php?Dataset=UWaveGestureLibraryAll)
3. [Dailysports](https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities) 
4. [WISDM](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
5. [GrabMyo](https://physionet.org/content/grabmyo/1.0.2/)


### Data Prepareation
We process each dataset individually by executing the corresponding `.py` files located in `data` directory. This process results in the formation of training and test `np.array` data, which are saved as `.pkl` files in `data/saved`. The samples are processed into the shape of (𝐿,𝐶).

For datasets comprising discrete sequences (UCI-HAR, Uwave and Dailysports), we directly use their original raw sequences as samples. For datasets comprising long-term, continuous signals (GrabMyo and WISDM), we apply sliding window techniques to segment these signals into
appropriately shaped samples (downsampling may be applied before window sliding). If the original dataset is not pre-divided into training and testing sets, a manual train-test split will be conducted. Information about the processed data can be found in `utils/setup_elements.py`. The saved data are **not preprocessed with normalization** due to the continual learning setup. Instead, we add a *non-trainable* input normalization layer before the encoder to do the sample-wise normalization. 

For convenience, we provide the processed data files for direct download. Please check the "Setup" part in the "Get Started" section.



### Setup
1. Download the processed data from [Google Drive](https://drive.google.com/drive/folders/1EFdD07myqmqHhRsjeQ83MdF8gHZXDWLR?usp=share_link). Put it into `data/saved` and unzip
   ```sh
   cd data/saved
   unzip <dataset>.zip
   ```
   You can also download the raw datasets and process the data with the corresponding python files.
2. Revise the following configurations to suit your device:
    * `resources` in `tune_hyper_params` in `experiment/tune_and_exp.py` (See [here](https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html) for details)
    * GPU numbers in the `.sh` files in `shell`.

### Run Experiment
for gamma=10
nohup sh shell/all_exp.sh &
for different gamma please change  TSACL\experiment\tune_config.py corresponding gamma value