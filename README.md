# HashtagGen


## Model

// Model Description

[comment]: <> (### The code will be released soon.)


## Usage

step 1： download requirements

```
conda create -n topic python=3.6
pip install -r requirements.txt
source activate topic
```

step 2: train/test/eval model

1. train the model
```
python -u run.py --num_gpus=1 --bert_config_file=./bert/sample/bert_config.json
```
2. test the model
```
python -u run.py --mode=test --init_checkpoint=checkpoint_2022-01-20-16-21-22 --checkpoint_file=best-0 --num_gpus=1 --coverage=false --use_pointer=false
# you can replace 'checkpoint_2022-01-20-16-21-22' with your own training checkpoint
```
3. eval the model
```
python -u run.py --mode=eval --init_checkpoint=checkpoint_2022-01-20-16-21-22 --checkpoint_file=best-0 --num_gpus=1 --coverage=false --use_pointer=false
# you can replace 'checkpoint_2022-01-20-16-21-22' with your own training checkpoint
```
4. configuration
```
1) `./bert/[sample|topic|topic_ltp]/bert_config.json` gives the train config files, you can follow our configurations.
2) `./bert/[sample|topic|topic_ltp]/vocab.txt` gives the bert vocabulary files
3) you can read `./run.py` to get more usage of our code.
```


## DataSet

We construct a Chinese large-scaletopic hashtag generation dataset (WHG) containing multiple areas from Weibo. If you want to acquire the  WHG \& THG corpus, please contact us, you will download the [application form](https://drive.google.com/file/d/1bZixZRQ8ypk-ZJByNl-juqyYR6NWiqBk/view?usp=sharing) and fill, then fax or e-mail. Contacts: Qianren Mao ( maoqr@act.buaa.edu.cn, cs: qianrenmao@gmail.com )




### Preview

Here is [an example of dataset](data):
```
weibo:
src: 天猫2017年双11成交额在今日零时40分20秒左右时突破500亿元。亿邦动力网注意到，2016年凌晨2点钟时，天猫双11成交额达到486亿元。
dst: 2017天猫双11
twitter:
src: former pl ams2 la reina adams credits her time with peo eis in her development as a leader . talent management is one of ms. smiths key priorities as peo . usa as c us army army acquisition
dst: talent management
```


Table 1: Data of WHG

WeiBo: WHG Dataset|Train|Dev|Test
-------|-----|---|----
Count |312,762| 2,000| 2,000
AvgSourceLen (+W) |75.1| 75.3 |75.6
CovSourceLen(95%)(+W) |141| 137 |145
AvgTargetLen(+W) |54.2 |4.2| 4.2
CovTargetLen(95%)(+W) |8 |8 |8

Table 2: Data of THG

Twitter: THG Dataset|Train|Dev|Test
-------|-----|---|----
Count |222,709| 2,000 | 2,000
AvgSourceLen  |23.5 |23.8 |23.5
CovSourceLen(95%)| 46 |47 |46
AvgTargetLen|10.1| 10.0 |10.0
CovTargetLen(95%)| 30 |30| 30


## Cite
@article{MAO2022109581,
title = {Attend and select: A segment selective transformer for microblog hashtag generation},
journal = {Knowledge-Based Systems},
pages = {109581},
year = {2022},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2022.109581},
url = {https://www.sciencedirect.com/science/article/pii/S0950705122007973},
author = {Qianren Mao and Xi Li and Bang Liu and Shu Guo and Peng Hao and Jianxin Li and Lihong Wang},
}
