from torch import nn

class DotProdNB(nn.Module):
    def __init__(self, nf, ny):
        super(DotProdNB).__init__()
        self.w = nn.Embedding(nf + 1, 1, padding_idx=0)
        self.w.weight.data_uniform(-0.1, 0.1)
        self.r = nn.Embedding(nf + 1, ny)

    def forward(self, feat_idx, feat_cnt, sz):
        w = self.w(feat_idx)
        r = self.r(feat_idx)
        x = ((w + 0.4) * r / 10).sum(1)
        return F.softmax(x)
