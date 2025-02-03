# MONKEY_CHALLENGE
![Screenshot 2025-02-03 at 2 56 48 PM](https://github.com/user-attachments/assets/f1f7708f-b948-4aad-b5de-c3222e1aac22)  
### Data cleaning  
As shown above, the overall workflow starts with data cleaning, in the cleaning phase, the first step is to [relabel all the cells with the StarDist model](predict.py).  
We used the pre-trained StarDist 2D_versatile_he model for a more precise monocyte/lymphocyte coordinate. It also allowed us to label the other cells outside the challenge target (simply named them 'other').  
With the StarDist model segmenting all the cells, we then tried a few ways to balance the amount of data for monocytes, lymphocytes, and others. After several approaches, we found the best dataset for training was when we doubled the data for monocytes.  
### Detection model  
For the detection model, we used the yolov11 model as our main detection model.   
The training code is shown [here](yolov11.py).  
We tested a few ways to train the Yolo model, including classification models for the three classes, monocyte/lymphocyte/other, or for only two classes, inflammatory cells/other. The results show that the precision of the two models is largely affected by the case, here we picked the three-class classification model (we named it the multiclass model) for the final test. Due to the size limit, the weight is uploaded [here](https://drive.google.com/drive/u/0/folders/1MayaHw4q85KfHbdaaaCU40cRQPV45Jjx).  
### Classification model  
After training a well-performed Yolo model, we still found it difficult to lower the false positive cells, so we added one more layer of classification model trained by the ground truth data, we used the [efficientnet model](train_efficientnet.py) to train the classification model for each cell. The classification model weights are uploaded [here](https://drive.google.com/drive/u/0/folders/1MayaHw4q85KfHbdaaaCU40cRQPV45Jjx). Class 0 is the classifier for lymphocytes, and class 1 for monocytes.
  
