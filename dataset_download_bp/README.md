## Dataset Download
`Dataset download` library is for downloading the CESM dataset and preprocessing it by segmenting only region of interest. It also splits the dataset into train and test data

If you would like to use your own data, please have the following schema
```
For vision finetuning, place the images in 'data\train_test_split_images` directory

data\train_test_split_images\train\
------------------------------------image_01.png
------------------------------------image_02.png
data\train_test_split_images\test\
------------------------------------image_11.png
------------------------------------image_12.png

For NLP model finetuning, place the data in 'data\annotation'  directory. The csv file should contain annotation notes from physician

data\annotation\training.csv : Contains training annotation notes by physican
data\annotation\testing.csv : Contains testing annotation notes by physican

```