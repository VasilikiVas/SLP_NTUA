import math
import sys
import torch.nn as nn
import torch

def progress(loss, epoch, batch, batch_size, dataset_size):
    """
    Print the progress of the training for each epoch
    """
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()


def train_dataset(_epoch, dataloader, model, loss_function, optimizer):
    # IMPORTANT: switch to train mode
    # enable regularization layers, such as Dropout
    model.train()
    running_loss = 0.0

    # obtain the model's device ID
    device = next(model.parameters()).device

    for index, batch in enumerate(dataloader, 1):
        # get the inputs (batch)
        inputs, labels, lengths, bows = batch
        # move the batch tensors to the right device
        inputs = inputs.to(device) # EX9
        lengths = lengths.to(device) # EX9
        labels = labels.to(device) # EX9
        bows = bows.to(device) # Lab3.6.1
        
        # Step 1 - zero the gradients
        # Remember that PyTorch accumulates gradients.
        # We need to clear them out before each batch!
        optimizer.zero_grad() # EX9

        # Step 2 - forward pass: y' = model(x)
        #import ipdb; ipdb.set_trace()
        ypred = model(inputs, lengths, bows) # EX9
        '''ypred, scores = model(inputs, lengths, bows) #lab3.5'''

        # Optimize labels to be compatible with each criterion
        if str(loss_function) == "BCEWithLogitsLoss()":  # EX9
           opt_labels = torch.nn.functional.one_hot(labels, 2).float()  # EX9
        else:
           opt_labels = labels  # EX9
        loss = loss_function(ypred, opt_labels) # EX9

        # Step 4 - backward pass: compute gradient wrt model parameters
        loss.backward() # EX9

        # Step 5 - update weights
        optimizer.step() # EX9

        running_loss += loss.data.item()

        # print statistics
        progress(loss=loss.data.item(),
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))

    return running_loss / index


def eval_dataset(dataloader, model, loss_function):
    # IMPORTANT: switch to eval mode
    # disable regularization layers, such as Dropout

    model.eval()
    running_loss = 0.0

    y_pred = []  # the predicted labels
    y = []  # the gold labels
    
    #Only for question 5
    #y_post = [] # the posteriors
    #y_scores = [] # the scores

    # obtain the model's device ID
    device = next(model.parameters()).device

    # IMPORTANT: in evaluation mode, we don't want to keep the gradients
    # so we do everything under torch.no_grad()
    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
            # get the inputs (batch)
            inputs, labels, lengths, bows = batch

            # Step 1 - move the batch tensors to the right device
            inputs = inputs.to(device) # EX9
            labels = labels.to(device) # EX9
            lengths = lengths.to(device) # EX9
            bows = bows.to(device)

            # Step 2 - forward pass: y' = model(x)
            ypred = model(inputs, lengths, bows) # EX9
            '''ypred, scores = model(inputs, lengths, bows) # lab3.5'''
            
            # Step 3 - compute loss.
            # We compute the loss only for inspection (compare train/test loss)
            # because we do not actually backpropagate in test time
            if str(loss_function) == "BCEWithLogitsLoss()":  # EX9
               opt_labels = torch.nn.functional.one_hot(labels, 2).float()  # EX9
            else:
               opt_labels = labels  # EX9
            loss = loss_function(ypred, opt_labels) # EX9

            # Step 4 - make predictions (class = argmax of posteriors)
            arg_max_post = torch.argmax(ypred, 1) # EX9

            # Step 5 - collect the predictions, gold labels and batch loss
            y_pred.append(arg_max_post.cpu().numpy()) # EX9
            y.append(labels.cpu().numpy()) # EX9
            
            '''y_post.append(ypred.cpu().numpy()) # Lab3.5.2
            y_scores.append(scores.cpu().numpy()) # Lab3.5.2'''

            running_loss += loss.data.item()

    return running_loss / index, (y_pred, y)
    '''return running_loss / index, (y_pred, y), y_post, y_scores #lab3.5'''