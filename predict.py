# example of pix2pix gan for satellite to map image-to-image translation
from numpy import load
import numpy as np
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.python.keras.models import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU

from matplotlib import pyplot
import xyz_to_clmb as x2c
import D2C
import sys 
import atom_data


# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (2,2), strides=(2,2), padding='valid', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (2,2), strides=(2,2), padding='valid', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (2,2), strides=(2,2), padding='valid', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	'''
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='valid', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='valid', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	'''
	# patch output
	d = Conv2D(1, (2,2), padding='valid', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (2,2), strides=(1,1), padding='valid', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (2,2), strides=(1,1), padding='valid', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g

# define the standalone generator model
def define_generator(image_shape=(100,100,2)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	#e5 = define_encoder_block(e4, 512)
	#e6 = define_encoder_block(e5, 512)
	#e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (2,2), strides=(1,1), padding='valid', kernel_initializer=init)(e4)
	b = Activation('relu')(b)
	# decoder model
	#d1 = decoder_block(b, e7, 512)
	#d2 = decoder_block(d1, e6, 512)
	#d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(b, e4, 512)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(2, (2,2), strides=(1,1), padding='valid', kernel_initializer=init)(d7)
	out_image = Activation('linear')(g)
	# define model
	model = Model(in_image, out_image)
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

# load and prepare training images
def load_real_samples(reac, pro):

    X1, X2 = np.array([x2c.job(reac)]), np.array([x2c.job(pro)])

    # load compressed arrays
    #data = load(filename, allow_pickle = True).item()

    #pr, rea, ts, names = data['products'], data['reactants'], data['ts'], data['names']

    arr1 = np.stack((X2, X1), axis = 3)
    arr2 = np.zeros(arr1.shape)#np.stack((ts, ts), axis = 3)
    #arr2 = ts.reshape((-1, 100, 100, 1))
    print (arr1.shape, arr2.shape)

    return [arr1, arr2]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape, ra = 1):
        # unpack dataset
        trainA, trainB = dataset
        # choose random instances
        if ra:
                ix = randint(0, trainA.shape[0], n_samples)
        else:
                ix = range (n_samples)
        #ix = randint(0, trainA.shape[0], n_samples)
        # retrieve selected images
        X1, X2 = trainA[ix], trainB[ix]
        # generate 'real' class labels (1)
        y = ones((n_samples, patch_shape, patch_shape, 1))
        return [X1, X2], y


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

def get_atoms(a, b):
    g = open(a)
    lines = g.readlines()[2:]
    g.close()
    at = []
    cords = []
    for line in lines:
        k = line.strip().split()
        if len(k) < 4:
            continue
        at.append(k[0])
        cords.append(list(map(float, k[1:])))

    g = open(b)
    lines = g.readlines()[2:]
    g.close()
    at2 = []
    cords2 = []
    for line in lines:
        k = line.strip().split()
        if len(k) < 4:
            continue
        at2.append(k[0])
        cords2.append(list(map(float, k[1:])))

    # Take mean of reactant and product
    #cords = np.mean([np.array(cords), np.array(cords2)], axis = 0)
    return at, np.array(cords)

def get_D(X, atoms):
    arr = []
    a_d = atom_data.symbol_dict(sys.argv[0])
    for i in range (len(atoms)):
        arr.append([])
        for j in range (len(atoms)):
            if i != j:
                r = (a_d[atoms[i]] * a_d[atoms[j]]) / X[i][j]
            else:
                r = 0.0
            arr[-1].append(r)

    # take average to make symmetric
    for i in range (len(atoms)):
        for j in range (i + 1, len(atoms)):
            arr[i][j] = (arr[i][j] + arr[j][i])/2

    for i in range (len(atoms)):
        for j in range (i + 1):
            arr[i][j] = arr[j][i]

    return np.array(arr)

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=1, name = 'temp'):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1, 0)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)

	X_fakeB = X_fakeB[0].T[0]

	atoms, cords = get_atoms(sys.argv[1], sys.argv[2])  # i.e from reactant file

	print (X_fakeB.shape)

	D_matrix = get_D(X_fakeB, atoms)

	#print (D_matrix)

	#cords = cords + np.random.normal(0,1,len(cords)*3).reshape(cords.shape)

	#print(D_matrix.shape, cords.shape)


	new_cords = D2C.job(D_matrix, cords, atoms)

	print ('job is done! and written as', name + '_ts.xyz')

	g = open(name + '_ts.xyz', 'w')
	g.write(str(len(atoms)) + "\nGenerated by TS-GAN\n")
	for i in range (len(atoms)):
		g.write(atoms[i] + ' ' + ' '.join(list(map(str, new_cords[i]))) + '\n')
	g.close()
    

# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=1, n_batch=1):
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		#print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance
		if 1:#(i+1) % (bat_per_epo * 1) == 0:
			summarize_performance(i, g_model, dataset)

# load image data
reac, pro = sys.argv[1], sys.argv[2]

dataset = load_real_samples(reac, pro)
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
g_model.load_weights('./g_model.h5')
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# train model
train(d_model, g_model, gan_model, dataset)
