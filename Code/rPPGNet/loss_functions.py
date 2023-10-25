import torch

class Neg_Pearson(torch.nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson,self).__init__()
        return
    def forward(self, preds, labels):       # all variable operation
        
        loss = 0
        
        sum_x = torch.sum(preds[0])                # x
        sum_y = torch.sum(labels)               # y
        sum_xy = torch.sum(preds[0]*labels)        # xy
        sum_x2 = torch.sum(torch.pow(preds[0],2))  # x^2
        sum_y2 = torch.sum(torch.pow(labels,2)) # y^2
        N = preds.shape[1]
        pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))

        #if (pearson>=0).data.cpu().numpy():    # torch.cuda.ByteTensor -->  numpy
            #   loss += 1 - pearson
        #else:
            #   loss += 1 - torch.abs(pearson)
        
        loss += 1 - pearson
            
            
        #loss = loss/preds.shape[0]
        return loss


losses = {
    "Binary" : torch.nn.BCELoss(),
    "Negative Pearsons" : Neg_Pearson,
    "MSE" : torch.nn.MSELoss()
}