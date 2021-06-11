# HashtagGen


## Model

// Model Description

### The code will be released soon.


## Data

We construct a Chinese large-scaletopic hashtag generation dataset (WHG) containing multiple areas from Weibo. It can be download at [google drive](https://drive.google.com/open?id=1vcJcVXKbVZ0z2acLjH3-e-qCLvFGpies). We also construct a English dataset from Twitter(THG).

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
Count |312,762| 20,000| 20,000
AvgSourceLen (+W) |75.1| 75.3 |75.6
CovSourceLen(95%)(+W) |141| 137 |145
AvgTargetLen(+W) |54.2 |4.2| 4.2
CovTargetLen(95%)(+W) |8 |8 |8

Table 2: Data of THG

Twitter: THG Dataset|Train|Dev|Test
-------|-----|---|----
Count |204,039| 11,335| 11,336
AvgSourceLen  |23.5 |23.8 |23.5
CovSourceLen(95%)| 46 |47 |46
AvgTargetLen|10.1| 10.0 |10.0
CovTargetLen(95%)| 30 |30| 30


