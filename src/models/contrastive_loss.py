import torch
import wandb
import PIL

import matplotlib.pyplot as plt


class SigmoidClipLoss(torch.nn.Module):
    
    def __init__(self, t_prime_init, b_init):
        super().__init__()
        # self.t_prime = torch.nn.Parameter(torch.tensor([2.3026])) # I think they meant to say t=10, not t'=10
        # self.t_prime = torch.nn.Parameter(torch.tensor([10.0])) # Initializations from Sigmoid paper
        # self.b = torch.nn.Parameter(torch.tensor([-10.0])) # Initializations from Sigmoid paper
        #self.t_prime = torch.tensor([t_prime_init]).to("cuda") #torch.nn.Parameter(torch.tensor([t_prime_init]))
        #self.b = torch.tensor([b_init]).to("cuda") #torch.nn.Parameter(torch.tensor([b_init]))
        
        self.t_prime = torch.nn.Parameter(torch.tensor([t_prime_init]))
        self.b = torch.nn.Parameter(torch.tensor([b_init]))
        #self.t_prime = torch.tensor([t_prime_init]).to("cuda")
        #self.b = torch.tensor([b_init]).to("cuda")
        
    def forward(self, loc_month_emb, chelsa_emb, label=None, verbose=False, l_module=False, plot_learnability=False):
        batchsize = loc_month_emb.shape[0]

        loc_month_emb = loc_month_emb / loc_month_emb.norm(p=2, dim=-1, keepdim=True)
        chelsa_emb = chelsa_emb / chelsa_emb.norm(p=2, dim=-1, keepdim=True)
        #if verbose:
            #print("loc_month_emb",loc_month_emb.max(), loc_month_emb.min(), loc_month_emb.mean())
            #print("chelsa_emb",chelsa_emb.max(), chelsa_emb.min(), chelsa_emb.mean())

        # cosine similarity as logits
        if verbose:
            l = torch.matmul(chelsa_emb, loc_month_emb.t())
            print("l_matmul",l.max(), l.min(), l.mean(), l.dtype)
            l = l * self.t_prime.exp()
            print("l_exp",l.max(), l.min(), l.mean(), l.dtype)
            l = l + self.b
            print("l_b",l.max(), l.min(), l.mean(), l.dtype)
        logits = torch.matmul(chelsa_emb, loc_month_emb.t()) * self.t_prime.exp() + self.b
        if verbose:
            print("logits",logits.max(), logits.min(), logits.mean())

        # Create labels matrix
        if not label is None:
            labels = label
        else:
            # If we have no specified label, use the -1;1 eye
            labels = torch.eye(batchsize).to(loc_month_emb.device) * 2 - 1
        if verbose:
            print("labels",labels.max(), labels.min(), labels.mean())

        # l = -sum(log_sigmoid(labels * logits)) / n
        if verbose:
            x = - labels * logits
            print("x_mult",x.max(), x.min(), x.mean(), x.dtype)
            x = x.exp()
            print("x_exp",x.max(), x.min(), x.mean(), x.dtype)
            x = 1 / (1 + x)
            print("x_plusdiv",x.max(), x.min(), x.mean(), x.dtype)
            x = torch.log(x)
            print("x_log",x.max(), x.min(), x.mean())
            x = torch.sum(x)
            print("x_sum",x.max(), x.min(), x.mean())
            x = -x / batchsize
            print("x_bs",x.max(), x.min(), x.mean())
        loss_matrix = torch.log(1 / (1 + (- labels * logits).exp()))
        loss = - torch.sum(loss_matrix) / batchsize
        if l_module:
            l_module.log("train/diagonal loss", - torch.mean(torch.diagonal(loss_matrix)))
            if plot_learnability:
                # Plot heatmap
                sub_array = - loss_matrix[::10,::10]
                sub_array = torch.log(sub_array - sub_array.min() + 1e-5).detach().cpu()
                fig, ax = plt.subplots(figsize=(8, 8))
                im = ax.imshow(sub_array, cmap='hot', interpolation='nearest')
                fig.colorbar(im, ax=ax)
                fig.savefig("./temp_loss.png")
                img = wandb.Image(PIL.Image.open("./temp_loss.png"))
                l_module.logger.experiment.log({"train/val learnability": img})

                plt.close()
        if verbose:
            print("loss",loss.max(), loss.min(), loss.mean())

        return loss


class RegressSimilarityLoss(torch.nn.Module):
    """Get the CHELSA-similarity matrix (matmul divided by norms) of matrix"""

    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, loc_month_emb, chelsa_emb, similarity=None, verbose=False):

        if similarity is None:
            raise ValueError("Data must provide chelsa similarity matrix.")

        loc_month_emb = loc_month_emb / loc_month_emb.norm(dim=1, keepdim=True)
        chelsa_emb = chelsa_emb / chelsa_emb.norm(dim=1, keepdim=True)

        logits = loc_month_emb @ chelsa_emb.t()

        loss = self.loss_fn(logits, similarity)

        return loss


class SoftmaxClipLoss(torch.nn.Module):
    """Adapted from https://github.com/microsoft/satclip/blob/main/satclip/ model.py and loss.py"""

    def __init__(self, logit_scale_init):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * logit_scale_init)
        #self.logit_scale = (torch.ones([]) * logit_scale_init).to("cuda")

    def forward(self, loc_month_emb, chelsa_emb, similarity=None, verbose=False):

        # normalized features
        loc_month_emb = loc_month_emb / loc_month_emb.norm(dim=1, keepdim=True)
        chelsa_emb = chelsa_emb / chelsa_emb.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_loc_month = logit_scale * loc_month_emb @ chelsa_emb.t()
        logits_per_chelsa = logits_per_loc_month.t()

        # shape = [global_batch_size, global_batch_size]

        labels = torch.arange(len(logits_per_loc_month), device=logits_per_loc_month.device, dtype=torch.long)

        loss = (
            torch.nn.functional.cross_entropy(logits_per_loc_month, labels) +
            torch.nn.functional.cross_entropy(logits_per_chelsa, labels)
        ) / 2

        return loss


if __name__ == "__main__":
    import torch

    sgm =  SigmoidClipLoss(2.34, -10.)
    sft = SoftmaxClipLoss(2.659260036932778)

    loc_month_emb = torch.tensor([[1,0,0,0.0],[0,1,0,0],[0,0,1,0]])
    chelsa_emb = torch.tensor([[1,0,0,0.0],[0,1,0,0],[0,0,1,0]])

    print("--------- Identical")
    print("SFT:", sft(loc_month_emb, chelsa_emb))
    print("SGM:", sgm(loc_month_emb, chelsa_emb))


    loc_month_emb = torch.tensor([[0,1,0,0],[0,0,1,0],[1,0,0,0.0]])
    chelsa_emb = torch.tensor([[1,0,0,0.0],[0,1,0,0],[0,0,1,0]])

    print("--------- Complete missmatch")
    print("SFT:", sft(loc_month_emb, chelsa_emb))
    print("SGM:", sgm(loc_month_emb, chelsa_emb))


    loc_month_emb = torch.tensor([[0,0,1,0],[0,1,0,0],[1,0,0,0.0]])
    chelsa_emb = torch.tensor([[1,0,0,0.0],[0,1,0,0],[0,0,1,0]])

    print("--------- 1 correct")
    print("SFT:", sft(loc_month_emb, chelsa_emb))
    print("SGM:", sgm(loc_month_emb, chelsa_emb))


    loc_month_emb = torch.tensor([[2,1,-1,0.0],[2,1,-1,0],[2,1,-1,0]])
    chelsa_emb = torch.tensor([[1,0,0,0.0],[0,1,0,0],[0,0,1,0]])

    print("--------- constant")
    print("SFT:", sft(loc_month_emb, chelsa_emb))
    print("SGM:", sgm(loc_month_emb, chelsa_emb))