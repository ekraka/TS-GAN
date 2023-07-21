import numpy as np
import sys

from ts_gan import print_out
from ts_gan import calculators as cal
from ts_gan import predict

class TSGAN:
    """

  *******  *****          *****    *****  *       *
     *    *              *     *  *     * * *     *
     *    *              *        *     * *  *    *
     *     *****   ****  *  ****  ******* *   *   *
     *         *         *     *  *     * *    *  *
     *         *         *     *  *     * *     * *
     *    ******          *****   *     * *       *

    The implementation of generative adversarial 
    networks (GAN) for the prediction of the transition state (TS) 
    geometry based on cartesian coordinates of product and reactant.

    Developed and maintained by Malgorzata-Z Makos and Niraj Verma 

    https://github.com/ekraka/TS-GAN

    Copyright:
    Computational and Theoretical Chemistry Group (CATCO), 
    Department of Chemistry, Southern Methodist University 

    Cite as:
    M.Z. Makoś, N. Verma, E.C. Larson, M. Freindorf, and E. Kraka; 
    J. Chem. Phys. 155, 2021, Vol.155, Issue 2 
    https://doi.org/10.1063/5.0055094


    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^       
        Instantiate the TSGAN class and run the code
        tsgan = TSGAN(sys.argv[1], sys.argv[2])
        tsgan.run()
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    """

    def __init__(self, reac, pro):
        """
        Parameters:
            reac (str): Filename of the reactant xyz file.
            pro (str): Filename of the product xyz file.
        
        """
        self.reac = reac
        self.pro = pro
        self.dataset = None
        self.image_shape = None
        self.d_model = None
        self.g_model = None
        self.gan_model = None

    def load_dataset(self):
        """
        Loads the real samples for the reactants and products.
        """
        print_out.get_header()
        # load image data
        self.dataset = predict.load_real_samples(self.reac, self.pro)
        print('Loaded', self.dataset[0].shape, self.dataset[1].shape)
        # define input shape based on the loaded dataset
        self.image_shape = self.dataset[0].shape[1:]

    def define_models(self):
        """
        Initialize and define the discriminator and generator models.
        The generator model is loaded with pre-trained weights from a file.
        """
                
        # define the models
        self.d_model = predict.define_discriminator(self.image_shape)
        self.g_model = predict.define_generator(self.image_shape)
        # loads pre-trained weights from a file. 
        self.g_model.load_weights('./g_model.h5')

    def define_gan_model(self):
        # define the composite model
        self.gan_model = predict.define_gan(self.g_model, self.d_model, self.image_shape)

    def train_model(self):
        # train model
        predict.train(self.d_model, self.g_model, self.gan_model, self.dataset)

    def run(self):
        """
        Execute the TSGAN pipeline.
        """
        self.load_dataset()
        self.define_models()
        self.define_gan_model()
        self.train_model()






