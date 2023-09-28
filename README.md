# **_InfoNet_**: Missing Information Retrieval in Multi-Stream Sensing Systems


## Step 01: Clone this repository
```
git clone https://github.com/hidethyself/InfoNet.git
cd InfoNet
```

## Step 02: Create virtual environment
```
apt-get install python3-venv
python3 -m venv venv
source venv/bin/activate
```

## Step 03: Install requirements
```
pip install -r requirements.txt
```

## Step 04: Download the data form [here](https://drive.google.com/drive/folders/1YbOdBA8p-WI_FRT7ktbiex3TXhZS7igb?usp=sharing). Unzip the downladed file into the   `<data>` folder.
```
mkdir data
cd data
unzip <downloaded_file>.zip
```

### Note: A small subset of data is provided with the repository to test the pipeline. To train the whole model with the full dataset, please follow the link given above.

<br>

## Step 05: Create the full-rank feature, $`F`$

```
python feature.py --full_rank
```

## Step 06: Train Baseline **_SELDNet_**
```
pyhton train.py
```

## Step 07: Create low-rank feature set, $`\tilde{F}`$
- Update `feature_params` from <code>parameters/parameters.py</code> . For example, to create $`\tilde{F}`$ with $`75\%`$ of dopping `feature_params` should be like this:
    ```python
    feature_params = {
        "is_doping": True,
        "doping_pct": .75,
        "no_dopped_channel": 3
        }
    ```
- Run the following:
    ```
    python feature.py
    ```

## Step 08: Train **_InfoNet_**:
```
python -W ignore train_infonet.py
```
