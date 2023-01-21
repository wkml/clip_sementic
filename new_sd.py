import sys
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from utils.lr_scheduler import ConstantWarmupScheduler
from utils.loss import AsymmetricLoss_partial

from utils.transforms import get_train_test_set
from utils.metrics import AverageMeter, AveragePrecisionMeter, Compute_mAP_VOC2012
from model.new_sd import DenseCLIP
from utils.checkpoint import save_checkpoint

from tensorboardX import SummaryWriter
import logging
from config import arg_parse, logger, show_args

global best_prec
best_prec = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global best_prec
    args = arg_parse()

    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    file_path = 'exp/log/{}.log'.format(args.post)
    file_handler = logging.FileHandler(file_path)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Show Argument
    show_args(args)

    # Create dataloader
    logger.info("==> Creating dataloader...")
    train_data_dir = args.train_data
    test_data_dir = args.test_data
    train_list = args.train_list
    test_list = args.test_list
    train_loader, test_loader = get_train_test_set(train_data_dir, test_data_dir, train_list, test_list, args)
    with open(args.category_file, 'r') as load_category:
        category_map = json.load(load_category)
    logger.info("==> Done!\n")

    # load the network
    logger.info("==> Loading the network ...")

    model = DenseCLIP(args=args, classnames=category_map)
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.prompt_learner.parameters():
        p.requires_grad = True

    # criterion = AsymmetricLoss_partial(gamma_neg=2, gamma_pos=1, clip=0.05).to(device)
    criterion = nn.BCEWithLogitsLoss(reduce=True, size_average=True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, model.parameters()),
    #                             lr=args.lr,
    #                             momentum=args.momentum, 
    #                             weight_decay=args.weight_decay,
    #                             nesterov=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    scheduler = ConstantWarmupScheduler(optimizer, scheduler, 0, 1e-5)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_mAP']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    logger.info("==> Done!\n")

    args.evaluate = True
    if args.evaluate:
        with torch.no_grad():
            print("eval mode!!!!!!!!")
            validate(test_loader, model, criterion, args)
        return

    logger.info("Run Experiment...")
    writer = SummaryWriter('{}/{}'.format('exp/summary/', args.post))

    for epoch in range(args.start_epoch, args.epochs):
        # model train mode
        train(train_loader, model, criterion, optimizer, epoch, args, writer)

        # evaluate on validation set
        with torch.no_grad():
            # model eval mode
            mAP = validate(test_loader, model, criterion, args)
        
        scheduler.step()
        writer.add_scalar('mAP', mAP, epoch)
        
        # remember best prec@1 and save checkpoint
        is_best = mAP > best_prec
        best_prec = max(mAP, best_prec)

        save_checkpoint(args, {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_mAP': mAP,
        }, is_best)

        if is_best:
            logger.info('[Best] [Epoch {0}]: Best mAP is {1:.3f}'.format(epoch, best_prec))
            
    writer.close()

def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input, target = input.to(device), target.float().to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)
        losses.update(loss.data, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Lr {lr:.6f}'.format(
                         epoch, i, len(train_loader), 
                         batch_time=batch_time,
                         loss=losses,
                         lr=optimizer.state_dict()['param_groups'][0]['lr']))
    writer.add_scalar('Loss', losses.avg, epoch)
    writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

def validate(val_loader, model, criterion, args):
    apMeter = AveragePrecisionMeter()
    pred, losses, batch_time = [], AverageMeter(), AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.to(device), target.float().to(device)

        output = model(input)
        loss = criterion(output, target)
        losses.update(loss.data, input.size(0))

        # Change target to [0, 1]
        target[target < 0] = 0

        apMeter.add(output, target)
        pred.append(torch.cat((output, (target>0).float()), 1))

        # Log time of batch
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses))
            sys.stdout.flush()

    pred = torch.cat(pred, 0).cpu().clone().numpy()
    mAP = Compute_mAP_VOC2012(pred, args.num_classes)

    averageAP = apMeter.value().mean()
    OP, OR, OF1, CP, CR, CF1 = apMeter.overall()
    OP_K, OR_K, OF1_K, CP_K, CR_K, CF1_K = apMeter.overall_topk(3)

    # with open(args.category_file, 'r') as load_category:
    #     classnames = json.load(load_category)
    # base_mAP, novel_mAP, class_ap = apMeter.compute_ap(classnames)
    # with open(args.post + '_ap.txt', 'w') as f:
    #     f.write(str(class_ap))

    # logger.info("base_mAP:{base_mAP:.3f}\n".format(base_mAP=base_mAP))
    # logger.info("novel_mAP:{novel_mAP:.3f}\n".format(novel_mAP=novel_mAP))
    logger.info('[Test] mAP: {mAP:.3f}, averageAP: {averageAP:.3f}\n'
                '\t\t\t\t\t(Compute with all label) OP: {OP:.3f}, OR: {OR:.3f}, OF1: {OF1:.3f}, CP: {CP:.3f}, CR: {CR:.3f}, CF1:{CF1:.3f}\n'
                '\t\t\t\t\t(Compute with top-3 label) OP: {OP_K:.3f}, OR: {OR_K:.3f}, OF1: {OF1_K:.3f}, CP: {CP_K:.3f}, CR: {CR_K:.3f}, CF1: {CF1_K:.3f}'.format(
                mAP=mAP, averageAP=averageAP,
                OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1, OP_K=OP_K, OR_K=OR_K, OF1_K=OF1_K, CP_K=CP_K, CR_K=CR_K, CF1_K=CF1_K))

    return mAP


if __name__=="__main__":
    main()

