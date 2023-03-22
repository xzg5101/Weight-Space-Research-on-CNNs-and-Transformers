from transformer import TransformerModel
import torch
# TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)

TransformerModel(ntoken = 10, ninp = 200, nhead = 2, nhid = 200,nlayers= 2, dropout= 0.2).to(device = torch.device("cpu"))