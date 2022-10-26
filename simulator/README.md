# InterSim - Beta

support Python 3.x

### Install from source
```
python setup.py install
```

### Install Tinker on Ubuntu
```
sudo apt-get install python3-tk
```

### Install Tinker on Mac or Windows
see [install instructions](https://tkdocs.com/tutorial/install.html) from Tinker

### Dataset
In order to  facilitate debugging, we arbitrarily place two scenes in [test_data](https://www.jianguoyun.com/p/DSMlGMkQrs6xCRiMzuwD). You can download it and then place it in `../driving_simulator/`.
In addition, you also can download complete dataset in [Waymo Motion Prediction Dataset](https://waymo.com/open/download/).


## Run a demo
This demo code continuously loads scenarios from Waymo Motion Prediction Dataset.
Run `test.py` to see scenario rendered

### Customization
By changing the hyperparameters in `config.py`, you can customize your own simulation environment.

