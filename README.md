# Show, Tell, and Beyond

Starting from the paper of [Show and Tell](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Vinyals_Show_and_Tell_2015_CVPR_paper.html), we reproduced it under Pytorch framework and went beyond it by integrating with a pretained ResNet feature extractor.

Implementation Details:

We adapted some code from [here](https://www.kaggle.com/dipanjandas96/image-caption-resnet-transformerdecoder-pytorch/notebook) in the image proprocessing stage.

Then, we manually implemented the CNN + LSTM networks to train.
