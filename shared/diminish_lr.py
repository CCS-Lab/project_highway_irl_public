def diminish_lr(start_lr,end_lr,start_epoch,end_epoch,current_epoch):
    lr_interval = (start_lr - end_lr)/(end_epoch-start_epoch) #start decreasing lr after start_epoch
    # print("lr_interval",lr_interval)
    if (current_epoch > start_epoch) and (current_epoch <= end_epoch):
        lr = start_lr - lr_interval * (current_epoch - start_epoch)
    elif current_epoch > end_epoch:
        lr = end_lr
    else:
        lr = start_lr
        
    return lr