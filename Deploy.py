import sys
from keras import models, layers
from keras import backend as K
import tensorflow as tf
import time
import glob

from bident.StrawMLTools import DeployBident
from bident.NonMaxSuppression import MultiResHandler

print('done importing')

# python3 Deploy.py <file.hic> </dir/models> </dir/working> <stem> <res,> <norm>

MATRIX_WIDTH = 500
FILEPATH = sys.argv[1]
MODEL_DIR = sys.argv[2]
WORKING_DIR = sys.argv[3]
STEM = sys.argv[4]
RESOLUTIONS = sys.argv[5].split(",")
THRESHOLD = 0.8
#WORKING_DIR = "/home/alyssa/loop_caller/machine-learning-deployment/"
#CELL_LINE = 'tridentv1_results/test'#GM12878_intact_30' #'gm12878_intra_nofrag_30'
#FILEPATH = '/home/alyssa/loop_caller/machine-learning-deployment/intact_test.hic'
#'/mnt/HIC2000/HiSeq/GM12878_2020_combined_maps/GM12878_quadRE_SDSallconc_10.5.20/inter_30.hic'
#'/mnt/HIC2000/HiSeq/GM12878_2020_combined_maps/GM12878_intact_combined_18.7B_10.5.20/inter_30.hic'
#'/mnt/HIC2000/HiSeq/GM12878_2020_combined_maps/GM12878_MNase_0.1_10.5.20/inter_30.hic'
#'/mnt/HIC2000/HiSeq/GM12878_2020_combined_maps/GM12878_quadRE_noSDS_10.5.20/inter_30.hic'
#'/mnt/HIC2000/HiSeq/GM12878_2020_combined_maps/GM12878_intact_combined_18.7B_10.5.20/inter_30.hic'
#'/mnt/HIC0000/HiSeq/MboI_magic_biorep/inter_30.hic' #'/mnt/HIC0000/mega/IMR90/DRGN/inter_30.hic'
NORMALIZATION_TYPE = sys.argv[6] #"KR"
MAX_EXAMPLES_IN_RAM = 30
BATCH_SIZE = 30
NUM_STRAW_WORKERS = 10

class deploy_trident_features():
	def __init__(self):
		print('initialize deployment', flush=True)
		models = self.load_models()

		resolutions = []
		for res_string in RESOLUTIONS:
			resolutions.append(int(res_string))
		resolutions.sort()
		print("Using resolutions", resolutions, flush=True)

		loop_lists = []
		domain_lists = []
		stripe_lists = []
		loop_domains_lists = []
		loop_radii = []
		domain_radii = []
		for res in resolutions:
			start_time = time.time()
			loops, domains, stripes, loop_domains = self.deploy_at_res(models, res)

			runtime = time.time()-start_time
			print("Executed resolution {} in {} seconds".format(res, runtime))
			print("-----------------------------------------", flush=True)

			loop_lists.append(loops)
			domain_lists.append(domains)
			stripe_lists.append(stripes)
			loop_domains_lists.append(loop_domains)
			loop_radii.append(res)
			domain_radii.append(3*res)
		self.merge_lists(loop_lists, domain_lists, stripe_lists, loop_domains_lists,
						 loop_radii, domain_radii)

	def wbce(self, y_true, y_pred, weight1=400, weight0=1):
		y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
		y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
		logloss = -(y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0 )
		return K.mean( logloss, axis=-1)

	def load_models(self):
		model_list = []
		for file in glob.glob(MODEL_DIR+"/*.h5"):
			print("Using Model", file, flush=True)
			model_list.append(models.load_model(file,
						custom_objects={'wbce': self.wbce,
            			'leaky_relu': tf.nn.leaky_relu,
            			'LeakyReLU': layers.LeakyReLU()}))
		return model_list

	def deploy_at_res(self, models, res):
		outpath = WORKING_DIR +"/"+ STEM
		loop_file = outpath + '_loops_' + str(res) + '.bedpe'
		domain_file = outpath + '_domains_' + str(res) + '.bedpe'
		stripe_file = outpath + '_stripes_' + str(res) + '.bedpe'
		loop_domain_file = outpath + '_loop_domains_' + str(res) + '.bedpe'
		print('got to deployment')
		print("Doing resolution", res, "for", STEM, flush=True)
		DeployBident(models=models, batchSize=BATCH_SIZE,
				 numStrawWorkers=NUM_STRAW_WORKERS, filepath=FILEPATH,
				 resolution=res,
				 maxExamplesInRAM=MAX_EXAMPLES_IN_RAM, matrixWidth=MATRIX_WIDTH,
				 threshold=THRESHOLD,
				 out_files=[loop_file,domain_file,stripe_file,loop_domain_file],
				 useArithmeticMean=False,
				 norm=NORMALIZATION_TYPE,
				 numOutputChannels=3)
		return loop_file, domain_file, stripe_file, loop_domain_file

	def merge_lists(self, loop_lists, domain_lists, stripe_lists, loop_domain_lists, loop_radii, domain_radii):
		if len(loop_lists) > 1:
			outpath = WORKING_DIR + "/" + STEM
			MultiResHandler(loop_lists, outpath + '_loops_merged.bedpe', threshold=1e-10,
							radii=loop_radii)
			MultiResHandler(domain_lists, outpath + '_domains_merged.bedpe', threshold=1e-10,
							radii=domain_radii)
			MultiResHandler(stripe_lists, outpath + '_stripes_merged.bedpe', threshold=1e-10,
							radii=loop_radii)
			MultiResHandler(loop_domain_lists, outpath + '_loop_domains_merged.bedpe', threshold=1e-10,
							radii=domain_radii)

print('starting deployment', flush=True)
deploy_trident_features()
