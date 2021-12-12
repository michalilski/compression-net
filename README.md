### Installation
- Run `pip install -r requirements.txt`

### Dataset and model
- Download  `cn.zip` from https://drive.google.com/drive/folders/1_mECiBUYz2-CTaza3lDgsyL0jnFfcvxm?usp=sharing
- Extract to `data/` and `models/` folder in this path

### Model
- Adjust all settings in `settings.py` folder

### Training
- Run `python3 train.py`

### Tests
1.For visual test:
- run `python3 results.py visual-test` to test random image from test set
- run `python3 results.py visual-test path/to/image.png` for specific image
2. For metrics test run `python3 results.py test-set-metrics`
3. For training set entropy distribution results run `python3 results.py train-set-entropy`
