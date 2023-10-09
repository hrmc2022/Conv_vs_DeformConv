import torch


def train_loop(device, epoch, data_loader, model, optimizer, loss_fn):

    print("epoch:", epoch)
    running_loss = 0.0
    losses = {"Training": 0.0, "Evaluation": 0.0}
    
    for phase in ["Training", "Evaluation"]:
        for batch, (image, target) in enumerate(data_loader[phase]):
            #print("batch: {}".format(batch))
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase == 'Training'):
                image = image.to(device)
                # print(image.shape)
                target = target.to(device)
                #print(target.shape)
                
                
                out = model(image)
                loss = loss_fn(out, target)
                #print("loss: {}".format(loss))
                
                if phase == "Training":
                    loss.backward()
                    # print("out_shape:", out.shape)
                    optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(data_loader[phase])
        losses[phase] = epoch_loss
        print("{} loss:{}".format(phase, epoch_loss))

    return losses
