from predicate import recognize
from train import run_training, run


def fastrun():
    run()


def predict(target):
    recognize(target)


def train(lr=None, image_dir=None):
    run_training(lr, image_dir)