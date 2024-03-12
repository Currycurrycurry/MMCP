# MMCP

## Experiment Setting
### Python Setting
Recommended version:  
* python (>=3.7) 
* tensorflow (>=2.3)
* lightgbm (=2.3, for truthful bidder)

### Sample Data
You can download the data from the following link and put the files in /data/ipinyou/xxxx/.

Download link: 

Extraction code: 

### Data Preparation
First step: Update the root path in config/base_config.py according to your own absolute path.
```
root_path = '/root/MMCP/'
```

Second step: Preprocess the data and encode the features.
```
python src/util/processor.py
```

(Optional) Third step: Train a CTR baseline model (lightgbm) for the truthful bidder. The truthful bidder is used to simulate bidding and split the original training logs.
```
python src/util/truthful_bidder.py
```

## Run MMCP
Please first preprocess the data and run the following code to train MMCP.
```
sh ./src/scripts/public/train_mmcp.sh camp_num
sh ./src/scripts/private/train_mmcp.sh
```