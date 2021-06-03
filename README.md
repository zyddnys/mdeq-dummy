# Multiscale Deep Equilibrium Models for dummies
The code in the [original MDEQ repo](https://github.com/locuslab/mdeq) is somewhat hard to modifiy for other applications. So I simplified it to make it easy to use by simply filling some NotImplemented part with your code.

# Tutorial
A simple demo of cifar10 can be found in [models_cifar10.py](models_cifar10.py).
1. You can make a copy of `model.py` and change it to your model name.
2. You can find comments in the file instructing you to fill missing modules.
3. The final model can be imported by importing `MDEQModelYourModel` from the file, of course you can change its name.

# Acknowledgement
A lot of codes are directly copied from the original MDEQ repo.
