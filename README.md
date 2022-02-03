# Promoters

Implementation of *PromoterLCNN: A Light CNN-based Promoter Prediction and Classification model*.

For convenience, this repository also contains the wrapper classes developed in order to compare with other models: [IPromoter-BnCNN](https://doi.org/10.1093/bioinformatics/btaa609) and [pcPromoter-CNN](https://www.mdpi.com/2073-4425/11/12/1529) in particular.

## Requirements
Just run:
```bash
# Necessary ones
$ pip install tensorflow tensorflow-addons biopython scikit-learn numpy pandas
# For Jupyter and table making
$ pip install jupyterlab tabulate
```
## Weights

Get them [here](https://drive.google.com/file/d/1D1XOIAUDMv04sZUIvgdfBAgL75lW8AgW/view?usp=sharing) (PromoterLCNN only) or [here](https://drive.google.com/file/d/1awsszk6905sVzetdgcQe5kOVTv4n70up/view?usp=sharing) (PromoterLCNN plus mirrors of both IPromoter-BnCNN and pcPromoter-CNN), then unzip inside ```weights``` folder.

## How to use

### Running inference on DNA strings

Each class is defined inside the module `custom_models`, in order to use them, all you have to do is something like this: 

```python
from custom_models import PromoterLCNN, PromoterType
from pathlib import Path

# First load the model
model = PromoterLCNN("PromoterLCNN", # Name for the instance
                [
                    Path("/path/to/first/classifier"), # Promoter/Non-promoter
                    Path("/path/to/second/classifier") # Promoter class
                ]
)
# Sample data
data = [
    "TTACTCATGGTTTTCTCCTGTCAGGAACGTTCGGATGAAAATTGATCCTTTCCAAGCTTAGACCAGGATGGCGGGATGGGC",
    "ATGCCTGATAATGAGAACTGCTTTAGTAAACTACTTTGTATGCTGTCTGTCTTTCAAACCGACGCAGCTATTAACGCATGA"
]
# Then call the predict method
results = model.predict(data)
# Results is a numpy array of PromoterType
```

### Running tests on DNA strings

See the examples inside [Overview](./Overview.ipynb)

## Training

See the examples inside [Train](./Train.ipynb)