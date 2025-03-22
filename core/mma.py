import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import coords_grid


class CFP(nn.Module):
    def __init__(self, c_dim):
        super(CFP, self).__init__()
        self.self_corr = nn.Linear(c_dim, c_dim)

    def fetch_mask(self, self_corr, corr, thres=0.4):
        corr_mask = torch.max(corr, dim=-1)[0]
        confidence = torch.zeros_like(corr_mask)  
        confidence[corr_mask <= thres] = -100
        confidence = confidence.unsqueeze(1)  
        self_corr = self_corr + confidence  

        self_corr = torch.softmax(self_corr, dim=-1)
        corr_mask[corr_mask > thres] = 1.0
        return self_corr, corr_mask.unsqueeze(-1)

    def forward(self, inp=None, corr_sm=None, self_corr=None, thres=0.4):
        if self_corr is None:
            batch, ch, ht, wd = inp.shape
            inp = inp.reshape(batch, ch, ht * wd).permute(0, 2, 1).contiguous()
            inp = self.self_corr(inp)
            self_corr = (inp * (ch ** -0.5)) @ inp.transpose(1, 2)

        flow_attn, conf = self.fetch_mask(self_corr, corr_sm, thres=thres)

        return flow_attn, conf, self_corr


class MMA(nn.Module):
    def __init__(self, c_dim):
        super(MMA, self).__init__()
        # self.ofe = OFE(args)
        self.cfp = CFP(c_dim=c_dim)
        self.multi_scale = True
        if self.multi_scale:
            chnn_hid = 32
            self.level_corr = 3
            self.gamma = nn.Parameter(torch.zeros(1))
            dila_s = [4, 8, 16]  # 8, 16, 24, 32
            conv_s = [nn.Sequential(
                nn.Conv2d(2 * self.level_corr, chnn_hid, 3, dilation=ii, padding=ii),
                nn.ReLU(inplace=True))
                for ii in dila_s]
            self.conv_s = nn.ModuleList(conv_s)
            chnn_ic = 256
            self.conv_rd = nn.Sequential(
                nn.Conv2d(len(dila_s) * chnn_hid, chnn_ic, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(chnn_ic, 2, 3, 1, 1))
            print(' -- Using multi-scale correlations for init_flow --')
            print(f' -- Number of Scale: {self.level_corr} --')

    def forward(self, fmap1, corr_pyramid, inp):
        batch, ch, ht, wd = fmap1.shape
        corr_i = corr_pyramid[0]
        h_d, w_d = corr_i.shape[-2:]
        assert h_d == ht and w_d == wd
        corr_sm = torch.softmax(corr_i.reshape(batch, ht * wd, h_d * w_d), dim=-1)

        crds_d = coords_grid(batch, h_d, w_d, device=corr_sm.device).reshape(batch, h_d * w_d, 2)
        crds = coords_grid(batch, ht, wd, device=corr_sm.device).reshape(batch, ht * wd, 2)
        flo = (corr_sm @ crds_d) * (ht / h_d) - crds
        flow_attn, conf, self_corr = self.cfp(inp=inp, corr_sm=corr_sm)

        flo = conf * flo + (1 - conf) * (flow_attn @ flo)
        flo_0 = flo.reshape(batch, ht, wd, 2).permute(0, 3, 1, 2).contiguous()

        if self.multi_scale:
            flo_s = []
            for ii in range(self.level_corr):
                corr_i = corr_pyramid[ii + 1]
                h_d, w_d = corr_i.shape[-2:]
                corr_sm = torch.softmax(corr_i.view(batch, ht * wd, h_d * w_d), dim=-1)

                crds_d = coords_grid(batch, h_d, w_d, device=corr_sm.device).view(batch, h_d * w_d, 2)
                flo = torch.einsum('b s m, b m f -> b s f', corr_sm, crds_d) * (ht / h_d) - crds

                flow_attn, conf, _ = self.cfp(self_corr=self_corr, corr_sm=corr_sm, thres=0.4 * 0.8 ** (ii + 1))

                flo = conf * flo + (1 - conf) * (flow_attn @ flo)
                flo = flo.view(batch, ht, wd, 2).permute(0, 3, 1, 2).contiguous()

                flo_s.append(flo)
            flos = torch.cat(flo_s, dim=1)

            flo_s = []
            for conv in self.conv_s:
                flo = conv(flos)
                flo_s.append(flo)
            flos = torch.cat(flo_s, dim=1)
            flo = self.conv_rd(flos)

            flo_0 = flo_0 + self.gamma * flo

        return flo_0
