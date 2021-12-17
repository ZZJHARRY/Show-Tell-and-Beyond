# Show, Tell, and Beyond

Starting from the paper of [Show and Tell](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Vinyals_Show_and_Tell_2015_CVPR_paper.html), we reproduced it under Pytorch framework and went beyond it by integrating with a pretained ResNet feature extractor.

# Download dataset:

```
! pip install kaggle
```

Before running the codes bellow, you need to make sure that you are at the root directory. Then, you need to get the kaggle.json file from the Kaggle website.

[Here](https://www.analyticsvidhya.com/blog/2021/06/how-to-load-kaggle-datasets-directly-into-google-colab/) is the instruction of downloading Kaggle datasets.

[Here](https://www.kaggle.com/adityajn105/flickr8k) is the link to the Flickr8k dataset on Kaggle.

```
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download adityajn105/flickr8k
!unzip flickr8k.zip
```

**Alternatively**: You could directly download the Flickr8k dataset on Kaggle from the link above and upload the unzipped files (Images directory and captions.txt) to the root directory. However, this method may be slower.



# Implementation Details:

**preporcess.py**

We adapted some ideas from [here](https://www.kaggle.com/dipanjandas96/image-caption-resnet-transformerdecoder-pytorch/notebook) in the image proprocessing stage. To preprocess the images, we first resize all of the images to be the shape of 224 * 224, and then we normalize all of the resized images.

After that, we applied a ResNet pretrained on Imagenet as the feature extractor on the preprocess images.

**encoder_decoder.py**

Then, we manually implemented the CNN + LSTM (including teacher forcing) networks to train.
