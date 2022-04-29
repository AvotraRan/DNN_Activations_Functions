import utils
import argparse
from config import args

# Model
parser =argparse.ArgumentParser()
parser.add_argument('-ac',
                    '--activation', 
                    help='This is the name of the activation function',
                    required=True
                    )


parser.add_argument('-n',
                    '--num_epochs', 
                    help='This is the number of epochs',
                    type=int,
                    required=True)


parser.add_argument('-i',
                    '--torch_implement', 
                    help='This is the default activation function',
                    type=bool,
                    required=True)

mains_args=vars(parser.parse_args())

num_epochs= mains_args['num_epochs']

torch_implement= mains_args['torch_implement']

a_c= mains_args['activation']


utils.my_function(a_c,num_epochs=num_epochs, use_pytorch=torch_implement)




