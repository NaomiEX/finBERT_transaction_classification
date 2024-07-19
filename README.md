# finBERT_transaction_classification

## Setup
Create the following folder structure: 
```
mkdir -p models/language_model/finbertTRC2
```

Download the pre-trained finBert model into this directory and copy the config.json
```
cp config.json models/language_model/finbertTRC2
wget https://prosus-public.s3-eu-west-1.amazonaws.com/finbert/language-model/pytorch_model.bin -P models/language_model/finbertTRC2
```

Now you can run `notebooks/route4.ipynb`