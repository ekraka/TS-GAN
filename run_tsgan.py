import os
import sys 
import ts_gan.predict 
from ts_gan.TS_GAN import TSGAN

# Get the reac and pro file names from command-line arguments
reac_file = sys.argv[1]
pro_file = sys.argv[2]

# Instantiate the GAN class and run the code
tsgan = TSGAN(reac_file, pro_file)
tsgan.run()



