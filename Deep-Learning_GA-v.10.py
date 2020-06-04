
import tqdm
import numpy
import pandas
import scipy.spatial


# hyperparameters used only when convergence method is 'best_chr'
BEST_CHROMOSOME_CONVERGENCE_LIMIT = 10
BEST_CHROMOSOME_CONVERGENCE_COUNTER = 0


# hyperparameters used only when convergence method is 'percentage'
PERCENTAGE_CHROMOSOME_CONVERGENCE_LIMIT = 1e-2


