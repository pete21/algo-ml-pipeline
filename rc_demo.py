import pprint
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from reservoir_computing.modules import RC_model
from reservoir_computing.utils import compute_test_scores
from reservoir_computing.datasets import ClfLoader

np.random.seed(0) # For reproducibility



config = {}

# Hyperarameters of the reservoir
config['n_internal_units'] = 500        # size of the reservoir
config['spectral_radius'] = 0.59        # largest eigenvalue of the reservoir
config['leak'] = 0.6                    # amount of leakage in the reservoir state update (None or 1.0 --> no leakage)
config['connectivity'] = 0.25           # percentage of nonzero connections in the reservoir
config['input_scaling'] = 0.1           # scaling of the input weights
config['noise_level'] = 0.01            # noise in the reservoir state update
config['n_drop'] = 5                    # transient states to be dropped
config['bidir'] = True                  # if True, use bidirectional reservoir
config['circle'] = False                # use reservoir with circle topology

# Dimensionality reduction hyperparameters
config['dimred_method'] = 'tenpca'      # options: {None (no dimensionality reduction), 'pca', 'tenpca'}
config['n_dim'] = 75                    # number of resulting dimensions after the dimensionality reduction procedure

# Type of MTS representation
config['mts_rep'] = 'reservoir'         # MTS representation:  {'last', 'mean', 'output', 'reservoir'}
config['w_ridge_embedding'] = 10.0      # regularization parameter of the ridge regression

# Type of readout
config['readout_type'] = 'lin'          # readout used for classification: {'lin', 'mlp', 'svm'}
config['w_ridge'] = 5.0                 # regularization of the ridge regression readout

pprint.pprint(config)


Xtr, Ytr, Xte, Yte = ClfLoader().get_data('Japanese_Vowels')

print(Xtr.shape, Ytr.shape, Xte.shape, Yte.shape)
# print(Xtr)
# print(Ytr)
# print(Xte)
# print(Yte)

# One-hot encoding for labels
onehot_encoder = OneHotEncoder(sparse_output=False)
Ytr = onehot_encoder.fit_transform(Ytr)
Yte = onehot_encoder.transform(Yte)

print(Ytr.shape)
# print(Yte.shape)
print(Ytr)
# print(Yte)

classifier =  RC_model(**config)

print("Training the model...")
# Train the model
tr_time = classifier.fit(Xtr, Ytr) 

print("Computing predictions on test data...")
# Compute predictions on test data
pred_class = classifier.predict(Xte) 
accuracy, f1 = compute_test_scores(pred_class, Yte)
print(f"Accuracy = {accuracy:.3f}, F1 = {f1:.3f}")
