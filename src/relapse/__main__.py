import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",action="store_true",help="Train the model")
    parser.add_argument("-v","--validate",action="store_true",help="Validate the model")
    parser.add_argument("-w","--weights",type=str,help="Path to the weights file")
    args = parser.parse_args()

    if args.validate and args.train:
        raise ValueError("Cannot validate and train at the same time")
    if args.validate and args.weights is None:
        raise ValueError("Cannot validate without weights")
    if args.validate:
        from relapse.validate.validate import validate
        validate(args.weights)
    if args.train:
        from relapse.train.relapse_train import train
        train()