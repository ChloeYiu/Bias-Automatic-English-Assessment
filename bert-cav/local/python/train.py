import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from transformers import AdamW
from data_prep import get_data, get_mask_with_feature
import sys
import os
import argparse
from tools import AverageMeter, get_default_device
from models import BERTGrader, BERTFeatureGrader, BERTLReLUGrader

def makeDir (name, mustBeNew):
    try:
        os.makedirs (name)
    except OSError as e:
        if e.errno != 17:
            raise e
        else:
            if mustBeNew:
                sys.stderr.write (('Directory "%s" already exists.'
                    + ' Remove this directory to proceed.\n') % name)
                sys.exit (100)
            else:
                # The directory already exists; no big deal.
                pass

def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=25):
    '''
    Run one train epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for i, (id, mask, target) in enumerate(train_loader):

        id = id.to(device)
        mask = mask.to(device)
        target = target.to(device)

        # Forward pass
        logits = model(id, mask)
        loss = criterion(logits, target)

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), id.size(0))

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, i, len(train_loader),
                      loss=losses))

def eval(val_loader, model, criterion, device):
    '''
    Run evaluation
    '''
    losses = AverageMeter()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (id, mask, target) in enumerate(val_loader):

            id = id.to(device)
            mask = mask.to(device)
            target = target.to(device)

            # Forward pass
            logits = model(id, mask)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            losses.update(loss.item(), id.size(0))

    print('Test\t Loss ({loss.avg:.4f})\n'.format(
              loss=losses))

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--OUT', type=str, help='Specify output th file')
    commandLineParser.add_argument('--RESPONSES', type=str, help='responses text file')
    commandLineParser.add_argument('--GRADES', type=str, help='scores text file')
    commandLineParser.add_argument('--FEATURE', type=str, default='', help='feature text file')
    commandLineParser.add_argument('--activation_fn', type=str, default="relu", help="Use LeakyReLU in the model")
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=3, help="Specify epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.00001, help="Specify learning rate")
    commandLineParser.add_argument('--sch', type=int, default=10, help="Specify scheduler param")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    commandLineParser.add_argument('--part', type=int, default=3, help="Specify part of exam")
    commandLineParser.add_argument('--val_size', type=int, default=500, help="Specify validation set size")
    commandLineParser.add_argument('--feature_size', type=int, default=356, help="Specify feature size")


    args = commandLineParser.parse_args()
    out_file = args.OUT
    responses_file = args.RESPONSES
    grades_file = args.GRADES
    feature_file = args.FEATURE
    batch_size = args.B
    epochs = args.epochs
    lr = args.lr
    sch = args.sch
    seed = args.seed
    part = args.part
    val_size = args.val_size
    feature_size = args.feature_size
    activation_fn = args.activation_fn

    torch.manual_seed(seed)

    # Save the command run
    out_dir = os.path.dirname(out_file)
    makeDir (out_dir, False)
    cmd_dir = os.path.join('CMDs',out_dir)
    makeDir (cmd_dir, False)
    with open(cmd_dir + '/train.cmds', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    device = get_default_device()

    # Load the data
    input_ids, mask, labels, speaker_ids = get_data(responses_file, grades_file, part=part)

    # If feature file is provided, load the features
    if feature_file:
        speaker_ids, mask = get_mask_with_feature(feature_file, speaker_ids, mask, feature_size)

    # split into training and validation sets
    input_ids_val = input_ids[:val_size]
    mask_val = mask[:val_size]
    labels_val = labels[:val_size]

    input_ids_train = input_ids[val_size:]
    mask_train = mask[val_size:]
    labels_train = labels[val_size:]

    # Use dataloader to handle batches
    train_ds = TensorDataset(input_ids_train, mask_train, labels_train)
    val_ds = TensorDataset(input_ids_val, mask_val, labels_val)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # initialise grader
    if feature_file:
        print("Using BERTFeatureGrader with feature size:", feature_size)
        model = BERTFeatureGrader(feature_size=feature_size)
    elif activation_fn == "lrelu":
        print("Using BERTLReLUGrader")
        model = BERTLReLUGrader()
    else:
        print("Using BERTGrader")
        model = BERTGrader()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[sch])

    # Criterion
    criterion = torch.nn.MSELoss(reduction='mean')

    # Train
    for epoch in range(epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_dl, model, criterion, optimizer, epoch, device)
        scheduler.step()

        # evaluate as we go along
        eval(val_dl, model, criterion, device)

    # Save the trained model
    state = model.state_dict()
    print("Saving model state to:", out_file)
    torch.save(state, out_file)
