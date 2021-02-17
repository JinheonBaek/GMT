import argparse

class Parser:

    def __init__(self):

        self.parser = argparse.ArgumentParser(description='Graph Multiset Transformer')
        self.parser.add_argument('--type', type=str, required=True)

        self.set_arguments()

    def set_arguments(self):

        self.parser.add_argument('--data', default='DD', type=str,
                            choices=['DD', 'PTC_MR', 'NCI1', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'MUTAG', 'COLLAB', 'ENZYMES'],
                            help='dataset type')
        self.parser.add_argument('--conv', default='GCN', type=str,
                            choices=['GCN', 'GIN'],
                            help='message-passing function type')
        self.parser.add_argument('--seed', type=int, default=42, help='seed')

        self.parser.add_argument('--num-hidden', type=int, default=128, help='hidden size')
        self.parser.add_argument('--num-heads', type=int, default=1, help='attention head size')

        self.parser.add_argument('--batch-size', default=128, type=int, help='train batch size')
        self.parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
        self.parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay')
        self.parser.add_argument("--grad-norm", type=float, default=1.0)
        self.parser.add_argument("--dropout", type=float, default=0.5)

        self.parser.add_argument('--pooling-ratio', type=float, default=0.25, help='pooling ratio')

        self.parser.add_argument('--num-epochs', default=500, type=int, help='train epochs number')
        self.parser.add_argument("--gpu", type=int, default=-1)
        self.parser.add_argument('--patience', type=int, default=50, help='patience for earlystopping')

    def parse(self):

        args, unparsed  = self.parser.parse_known_args()
        
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        
        return args