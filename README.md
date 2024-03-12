# MMCP

## Experiment Setting
### Python Setting
Recommended version:  
* python (>=3.7) 
* tensorflow (>=2.3)
* lightgbm (=2.3, for truthful bidder)

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

## Reference

Please cite our work if you find our code/paper is useful to your work:
```
@inproceedings{10.1145/3616855.3635838,
author = {Huang, Jiani and Zheng, Zhenzhe and Kang, Yanrong and Wang, Zixiao},
title = {From Second to First: Mixed Censored Multi-Task Learning for Winning Price Prediction},
year = {2024},
isbn = {9798400703713},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3616855.3635838},
doi = {10.1145/3616855.3635838},
abstract = {A transformation from second-price auctions (SPA) to first-price auctions (FPA) has been observed in online advertising. The consequential coexistence of mixed FPA and SPA auction types has further led to the problem of mixed censorship, making bid landscape forecasting, the prerequisite for bid shading, more difficult. Our key insight is that the winning price (WP) under SPA can be effectively transferred to FPA scenarios if they share similar user groups, advertisers, and bidding environments. The full utilization of winning price under mixed censorship can effectively alleviate the FPA censorship problem and improve the performance of winning price prediction (aka. bid landscape forecasting). In this work, we propose a Multi-task Mixed Censorship Predictor (MMCP) that utilizes multi-task learning (MTL) to leverage the WP under SPA as supervised information for FPA. A Double-gate Mixture-of-Experts architecture has been proposed to alleviate the negative transfer problem of multi-task learning in our context. Furthermore, several auxiliary modules including the first-second mapping module and adaptive censorship loss function have been introduced to integrate MTL and winning price prediction. Extensive experiments on two real-world datasets demonstrate the superior performance of the proposed MMCP compared with other state-of-the-art FPA models under various performance metrics. The implementation of the code is available on github (https://github.com/Currycurrycurry/MMCP/).},
booktitle = {Proceedings of the 17th ACM International Conference on Web Search and Data Mining},
pages = {295–303},
numpages = {9},
keywords = {bid landscape forecasting, bid shading, market price modeling, mixed censorship, multi-task learning, real-time bidding, winning probability estimation},
location = {<conf-loc>, <city>Merida</city>, <country>Mexico</country>, </conf-loc>},
series = {WSDM '24}
}
```
