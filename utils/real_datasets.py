
from sklearn.datasets import fetch_openml
import pandas as pd
from scipy import io 
import shap 
import numpy as np 
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from ucimlrepo import fetch_ucirepo 
from sklearn.impute import SimpleImputer

def load_dataset(name):
    X, y = load_dataset_basic(name)

    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X)
    X_imputed = imputer.transform(X) 

    return X_imputed, y


    
def load_dataset_basic(name):
    """
    Load dataset by name.

    Parameters:
        name: Name of the dataset

    Returns:
        X, y: Feature matrix and labels
    """
    down_sample = False
    down_sample_rate = 1 
    imbalance = False
    if name == "madelon":
        dataset = fetch_openml(name="madelon", version=1, as_frame=True)

    elif name == "nomao":
        dataset = fetch_openml(name="nomao", version=1, as_frame=True)
        down_sample=True
        imbalance = True
        down_sample_rate = 0.3
    
    elif name == "waveform":
        dataset = fetch_openml(name="waveform-5000", version=1, as_frame=True)

    elif name == "steel":
        dataset = fetch_openml(name="steel-plates-fault", version=1, as_frame=True)

    elif name == "sonar":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
        dataset = pd.read_csv(url, header=None)
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        return X.values, y.values
    
    elif name == "ionosphere":
        ds = io.loadmat('data/ionosphere.mat')
        return ds["X"], ds["y"]

    elif name == "gas":
        data = pd.read_csv('data/gas.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values
    
    elif name == "breast_cancer_wisconsin": #classification
        breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
        # data (as pandas dataframes) 
        X = breast_cancer_wisconsin_diagnostic.data.features 
        y = breast_cancer_wisconsin_diagnostic.data.targets 
        return X.values, y.values
    
    elif name == "breast_cancer": #regression
        data = pd.read_csv('data/breastcancer.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values
    
    elif name == "pol": #regression
        data = pd.read_csv('data/pol.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values   
    
    elif name == "bike": #regression
        data = pd.read_csv('data/bike.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values 
    
    elif name == "autos": #regression
        data = pd.read_csv('data/autos.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values 
    
    elif name == "pumadyn32nm": #regression
        data = pd.read_csv('data/pumadyn32nm.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values

    elif name == "skillcraft": #regression
        data = pd.read_csv('data/skillcraft.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values
    
    elif name == "sml": #regression
        data = pd.read_csv('data/sml.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values
    
    elif name == "keggdirected": #regression
        data = pd.read_csv('data/keggdirected.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values
    
    elif name == "keggundirected": #regression
        data = pd.read_csv('data/keggundirected.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        X_down, _, y_down, _ = train_test_split(X.values, y.values, train_size=0.25, random_state=42)
        return X_down, y_down
    
    elif name == "parkinson": #regression
        data = pd.read_csv('data/parkinson.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values
    
    elif name == "crime":
        X, y = shap.datasets.communitiesandcrime()
        return X.values, y

    else:
        raise ValueError(f"Unknown dataset: {name}")

    X, y = dataset.data, dataset.target

    if down_sample:
        if imbalance:
            rus = RandomUnderSampler(random_state=42)
            X_downsampled, y_downsampled = rus.fit_resample(X, y)
        
        X, _, y, _ = train_test_split(X_downsampled, y_downsampled, train_size=down_sample_rate, random_state=42)

    return np.array(X), np.array(y)

