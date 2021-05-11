# First we load all the libraries
import pandas as pd

# We load the data

msnn = pd.read_csv("Experimental_results/Experiment5_MultivariableShallowNN_HyperparamSearchResults.csv")
print(msnn.shape)
print(msnn.head())

from tqdm import tqdm
for i in tqdm(range(10)):
    sleep(3)
    
tqdm.auto


