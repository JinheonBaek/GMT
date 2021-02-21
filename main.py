import argparse

def main(work_type_args):

    if work_type_args.type == 'classification_TU':
        from parsers.classification_TU import Parser
        from trainers.trainer_classification_TU import Trainer

    elif work_type_args.type == 'classification_OGB':
        from parsers.classification_OGB import Parser
        from trainers.trainer_classification_OGB import Trainer

    elif work_type_args.type == 'reconstruction_ZINC':
        from parsers.reconstruction_ZINC import Parser
        from trainers.trainer_reconstruction_ZINC import Trainer
        
    else:
        raise ValueError("Work Type Name <{}> is Unknown".format(work_type_args.type))

    args = Parser().parse()
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':

    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, required=True)

    main(work_type_parser.parse_known_args()[0])