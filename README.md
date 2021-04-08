# TS-GAN
Implementation of conditional GAN in the predition of the transition state (TS) geometry based on the Cartesian coordinates of product and reactant. 

### Prerequisites:
* Python 3.8.0
* Tensorflow 2.2.0


### Instalation:
        git clone 


### Prediction
In order to predict the TS guess structure, make sure the `g_mode.h5` file is in to the same working directory as `xyz` files of reactant and product. 
The `g_mode.h5` can be foung in the `test_cases` folder depending on which reaction one is interested in.

    python predict.py reactant.xyz product.xyz
    
Prediction script will generate two files: `temp_ts.xyz` and `temp_mov.xyz`. First file show the final guess structure, while the second file show the movie on how the optimization took place. Before use, we recomend open both files in any visualization program and check whether the generated guess is the desired one. In some cases choosing coordinates from `temp_mov.xyz` might be a better option for the further TS optimization. 

### Training
To train the model on your own data, convert `xyz` files of reactants, transition states, and products into the Coulomb matrices (CMs) using `xyz_to_clmb.py`. Ensure data is in npy format and CMs of reactants, transition states, products are aligned.

    python train.py 

### If used, please cite. 

