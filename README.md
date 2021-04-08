# TS-GAN
Implementation of conditional GAN in the predition of the transition state (TS) geometry based on the Cartesian coordinates of product and reactant. 

### Prerequisites:
* Python 3.8.0
* Tensorflow 2.2.0


### Instalation:
Go to the working directory:

        git clone https://github.com/ekraka/TS-GAN.git


### Prediction
In order to predict the TS guess structure, make sure the `g_mode.h5` file is in to the same working directory as `xyz` files of reactant and product. 
The `g_mode.h5` can be foung in the `test_cases` folder depending on which reaction one is interested in.

    python predict.py reactant.xyz product.xyz
    
Prediction script will generate two files: `temp_ts.xyz` and `temp_mov.xyz`. First file show the final guess structure, while the second file show the movie on how the optimization took place. Before use, we recomend open both files in any visualization program and check whether the generated guess is the desired one. In some cases choosing coordinates from `temp_mov.xyz` might be a better option for the further TS optimization. 

### Training
To train the model on your own data, convert `xyz` files of reactants, transition states, and products into the Coulomb matrices (CMs) using `xyz_to_clmb.py`. Ensure data is in npy format and CMs of reactants, transition states, products are aligned. Create an empty folder `temp` in your working directory then run:

    python train.py 

During the training process, the model will save waigths of discriminator and generator as `d_model.h5` and `g_model.h5`, respectivelly. While the random real and fake samples of the CMs will be saved in the `temp` folder. 

### Test
To calculate the root-mean-square deviation (RMSD) use:

        python align3D.py ts.xyz temp_ts.xyz


### If used, please cite. 

