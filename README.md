# TS-GAN
The implementation of conditional GAN for the prediction of the transition state (TS) geometry.
For more detils refer [here to our publication.](https://aip.scitation.org/doi/10.1063/5.0055094) 

### Prerequisites:
* Python 3.8.0
* Tensorflow 2.2.0


### Installation:
Go to the working directory:

        git clone https://github.com/ekraka/TS-GAN.git


### Prediction
To predict the TS guess structure, make sure the `g_mode.h5` file is in the same working directory as `xyz` files of reactant and product. 
The `g_mode.h5` can be found in the `test_cases` folder depending on which reaction one is interested in.

    python predict.py reactant.xyz product.xyz
    
Prediction script will generate two files: `temp_ts.xyz` and `temp_mov.xyz`. The first file shows the final guess structure, while the second file shows the movie on how the optimization took place. 

### Training
To train the model on your own data, convert `xyz` files of reactants, transition states, and products into the Coulomb matrices (CMs) using `xyz_to_clmb.py`. Ensure data is in `npy` format and CMs of reactants, transition states, products are aligned. Create an empty folder `temp` in your working directory, then run:

    python train.py 

During the training process, the model will save weights of discriminator and generator as `d_model.h5` and `g_model.h5`, respectively. While the random real and fake samples of the CMs are saved in the `temp` folder. 

### Test
To calculate the root-mean-square deviation (RMSD) use:

        python align3D.py ts.xyz temp_ts.xyz


### Cite as: 
J. Chem. Phys. 155, (2021), Vol.155, Issue 2; doi: 10.1063/5.0055094

       @article{TSGAN,
       doi = {10.1063/5.0055094},
       year = {2021},
       publisher = {{AIP} Publishing},
       volume = {155},
       number = {2},
       pages = {024116},
       author = {Ma{\l}gorzata Z. Mako{\'{s}} and Niraj Verma and Eric C. Larson and Marek Freindorf and Elfi Kraka},
       title = {Generative adversarial networks for transition state geometry prediction},
       journal = {J Chem Phys}
       }

