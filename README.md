# TS-GAN
Implementation of conditional GAN in the predition of the transition state (TS) geometry based on the Cartesian coordinates of product and reactant. 

### Prerequisites:
* Python 3.8.0
* Tensorflow 2.2.0

### Prediction
TS-GAN was trained on three different data sets: A) hydrogen migration reactions, B) isomerizarion reaction, and C) transition metal catalyzed reaction. 
Go to `test_cases` folder and copy `g_mode.h5` file to your working directory as `xyz` files of reactant and product.
  python predict.py reactant.xyz product.xyz

