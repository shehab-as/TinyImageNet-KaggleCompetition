## Assignment 3 of Machine Learning Course: miniature of ImageNet classification challenge.
Achieved Accuracy of <strong><i>83.30%</i></strong>, placed to 2nd rank.

#### Using Xception Model, loading pre-trained Imagenet weights and fine-tuning to achieve better validation accuracy.
#### 10% of the data used for validation, rounding up to 90,000 train images, and 10,000 validation images.
#### Re-structured dataset to the following stucture:

```
data/ 

   train/
      class1/
      class2/
      
   validation/
      class1/
      class2/
```

### Install dependencies before running the program

pip3 install -r requirements.txt (Python 3.5.2)