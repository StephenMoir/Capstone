import argparse
import json
import os
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

# imports the model in model.py by name
from model import BinaryClassifier

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryClassifier(model_info['input_features'], model_info['hidden_dim'], model_info['output_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model

# Gets training data in batches from the nntrain.csv file
#def _get_train_data_loader(batch_size, training_dir):
#    print("Get train data loader.")

#    train_data = pd.read_csv(os.path.join(training_dir, "nntrain.csv"), header=None, names=None)
#    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "nntrain.csv"), header=None, names=None)

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_x = torch.from_numpy(train_data.drop([0], axis=1).values).float()

    train_ds = torch.utils.data.TensorDataset(train_x, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

# Gets test data in batches from the nntest.csv file
def _get_test_data_loader(batch_size, training_dir):
    print("Get test data loader.")

    test_data = pd.read_csv(os.path.join(training_dir, "nntest.csv"), header=None, names=None)

    test_y = torch.from_numpy(test_data[[0]].values).float().squeeze()
    test_x = torch.from_numpy(test_data.drop([0], axis=1).values).float()

    test_ds = torch.utils.data.TensorDataset(test_x, test_y)

    return torch.utils.data.DataLoader(test_ds, batch_size=batch_size)


# Provided training function
def train(model, train_loader,  epochs, criterion, optimizer, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training. 
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    #train_losses = []
    
    # training loop is provided
    for epoch in range(1, epochs + 1):
        model.train() # Make sure that the model is in training mode.

        total_loss = 0

        for batch in train_loader:
            # get data
            batch_x, batch_y = batch

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # get predictions from model
            y_pred = model(batch_x)
            
            # perform backprop
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()
            
        
        #train_losses.append(total_loss/len(trainloader))

        print("Epoch: {}, Training Loss: {}".format(epoch, total_loss / len(train_loader)))
        
    #return train_losses 

# Provided training function
def test(model, test_loader, epochs, criterion, optimizer, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    test_loader  - The PyTorch DataLoader that should be used to check epoch overfitting.
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training. 
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    test_losses = []
    
    # training loop is provided
    for epoch in range(1, epochs + 1):
        model.train() # Make sure that the model is in training mode.

        test_loss = 0
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
                
            for tbatch in test_loader:
                # get data
                tbatch_x, tbatch_y = tbatch

                tbatch_x = tbatch_x.to(device)
                tbatch_y = tbatch_y.to(device)

                
    
                # get predictions from model
                ty_pred = model(tbatch_x)
            
                # perform backprop
                tloss = criterion(ty_pred, tbatch_y)
            
                test_loss += tloss.data.item()
        
 
    test_losses.append(test_loss/len(testloader))
    print("Epoch: {}, Test Loss    : {}".format(epoch, total_loss / len(train_loader)))
        
    return test_losses

## main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    ## TODO: Add args for the three model parameters: input_features, hidden_dim, output_dim
    # Model Parameters
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    
    parser.add_argument('--input_features', type=int, default=256, metavar='ID',
                        help='The Input dimensions (default: 256)')
    
    parser.add_argument('--hidden_dim', type=int, default=128, metavar='HD',
                        help='The Hidden dimensions (default: 128)')
    
    parser.add_argument('--output_dim', type=int, default=1, metavar='OD',
                        help='The Output dimensions (default: 1)')
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)
    
    # Load the test data.
    test_loader = _get_test_data_loader(args.batch_size, args.data_dir)

    # TYPICAL UPDATE SECTION
    
    ## Build the model by passing in the input params
    # To get params from the parser, call args.argument_name, ex. args.epochs or ards.hidden_dim
    # Don't forget to move your model .to(device) to move to GPU , if appropriate
    
    model = BinaryClassifier(args.input_features, args.hidden_dim, args.output_dim)
    model.to(device)

    ## TODO: Define an optimizer and loss function for training

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # Trains the model (given line of code, which calls the above training function)
    train(model, train_loader, args.epochs, criterion, optimizer, device)
    
    #pd.DataFrame(train_losses).to_csv(os.path.join(training_dir, 'training_loss.csv'))
    #pd.DataFrame(test_losses).to_csv(os.path.join(training_dir, 'test_loss.csv'))

    ## TODO: complete in the model_info by adding three argument names for model architecture
    # Keep the keys of this dictionary as they are 
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_features': args.input_features,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim,
        }
        torch.save(model_info, f)
        
    ## --- End of your code  --- ##
    

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
