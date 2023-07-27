# TS-GAN
The implementation of conditional GAN for the prediction of the transition state (TS) geometry based on cartesian coordinates of product and reactant.
For more information, please refer [here](https://aip.scitation.org/doi/10.1063/5.0055094) to our publication.

### Note from the developer
The updated version of the TS-GAN that uses the up-to-date TensorFlow 2.12 and Atomistic Simulation Environment (ASE) to speed up the code is available in the `updated` branch [here](https://aip.scitation.org/doi/10.1063/5.0055094). 


### Prerequisites:
* Python 3.8.16
* Tensorflow 2.2.0
* Numpy 1.20.0



### Installation:
Go to the working directory:

        git clone https://github.com/ekraka/TS-GAN.git


### Prediction
To predict the TS guess structure, make sure the `g_model.h5` file is in the same working directory as `xyz` files of reactant and product. 
The `g_model.h5` can be found in the `test_cases` folder depending on which reaction one is interested in.

    python predict.py reactant.xyz product.xyz
    
Prediction script will generate two files: `temp_ts.xyz` and `temp_mov.xyz`. The first file shows the final guess structure, while the second file shows the movie on how the optimization took place. 

### Training
To train the model on your own data, convert `xyz` files of reactants, transition states, and products into the Coulomb matrices (CMs) and store as numpy file. This can be done with the gen_data.py script which requires specific format for files. The reactnats, transition states and products should be kept in a single folder with the following name specification:

        Transition state: filename.xyz
        Reactant: filename_rev.xyz
        Product: filename_for.xyz

To create the numpy data, run the following command from this directory:

        python path_to_TS-GAN/gen_data.py
        
This will create a file called data.npy

To train through this data, create a prefered working directory and copy the data.npy file. Also, create an empty folder `temp` in this directory, then run:

    python train.py 

During the training process, the model will save weights of discriminator and generator as `d_model.h5` and `g_model.h5`, respectively. While the random real and fake samples of the CMs are saved in the `temp` folder. 

### Test
To calculate the root-mean-square deviation (RMSD) use:

        python align3D.py ts.xyz temp_ts.xyz


### Cite as: 
M.Z. Makoś, N. Verma, E.C. Larson, M. Freindorf, and E. Kraka; Generative Adversarial Networks for Transition State Geometry Prediction; J. Chem. Phys. 155, (2021), Vol.155, Issue 2; doi.org/10.1063/5.0055094

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

