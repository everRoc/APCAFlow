import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import coords_grid


class CFP(nn.Module):
    def __init__(self, c_dim):
        super(CFP, self).__init__()
        self.self_corr = nn.Linear(c_dim, c_dim)

    def fetch_mask(self, self_corr, corr, matching_mask=None, nomask=False, thres=0.4):
        corr_mask = torch.max(corr, dim=-1)[0]
        confidence = torch.zeros_like(corr_mask)
        if not nomask:
            # print("conf mask is comming!")
            if matching_mask is not None:
                confidence[matching_mask > 0.5] = -100
            else:
                confidence[corr_mask <= thres] = -100
        confidence = confidence.unsqueeze(1)  
        self_corr = self_corr + confidence  
        self_corr = torch.softmax(self_corr, dim=-1)

        if nomask:
            corr_mask = torch.zeros_like(corr_mask)
        elif matching_mask is not None:
            corr_mask[matching_mask < 0.5] = 1.0
        else:
            corr_mask[corr_mask > thres] = 1.0
        return self_corr, corr_mask.unsqueeze(-1)

    def forward(self, inp=None, corr_sm=None, self_corr=None, matching_mask=None,
                nomask=False, thres=0.4):
        if self_corr is None:
            batch, ch, ht, wd = inp.shape
            inp = inp.reshape(batch, ch, ht * wd).permute(0, 2, 1).contiguous()
            inp = self.self_corr(inp)
            self_corr = (inp * (ch ** -0.5)) @ inp.transpose(1, 2)

        flow_attn, conf = self.fetch_mask(self_corr, corr_sm, matching_mask=matching_mask,
                                          nomask=nomask, thres=thres)

        return flow_attn, conf, self_corr


class MMA(nn.Module):
    def __init__(self, c_dim, matchmask=False, nomask=False):
        super(MMA, self).__init__()
        # self.ofe = OFE(args)
        self.matchmask = matchmask
        self.nomask = nomask
        self.cfp = CFP(c_dim=c_dim)
        self.multi_scale = True
        if self.multi_scale:
            chnn_hid = 32
            self.level_corr = 2
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

    def fetch_matching_mask(self, corr, thres=0.4):
        b = corr.shape[0]
        match12, match_idx12 = corr.max(dim=2)  # (N, fH*fW)
        match21, match_idx21 = corr.max(dim=1)

        for b_idx in range(b):
            match21_b = match21[b_idx, :]
            match_idx12_b = match_idx12[b_idx, :]
            match21[b_idx, :] = match21_b[match_idx12_b]
        matching_mask = ((match12 - match21) != 0.0).float()  # (N, fH*fW)
        return matching_mask

    def forward(self, fmap1, corr_pyramid, inp):
        batch, ch, ht, wd = fmap1.shape
        corr_i = corr_pyramid[0]
        h_d, w_d = corr_i.shape[-2:]
        assert h_d == ht and w_d == wd
        corr_sm = corr_i.reshape(batch, ht * wd, h_d * w_d)
        matching_mask = None
        if (not self.nomask) and self.matchmask:
            print("matchmask is comming!!!!!!!")
            matching_mask = self.fetch_matching_mask(corr_sm)
        corr_sm = torch.softmax(corr_sm, dim=-1)
        # corr_sm = torch.softmax(corr_i.reshape(batch, ht * wd, h_d * w_d), dim=-1)

        # crds_d = coords_grid(batch, h_d, w_d, device=corr_sm.device).view(batch, 2, h_d*w_d).permute(0, 2, 1)
        crds = coords_grid(batch, ht, wd, device=corr_sm.device).view(batch, 2, ht*wd).permute(0, 2, 1)
        flo = (corr_sm @ crds) - crds
        flo_match = flo.reshape(batch, ht, wd, 2).permute(0, 3, 1, 2).contiguous()
        flow_attn, conf, self_corr = self.cfp(inp=inp, corr_sm=corr_sm, matching_mask=matching_mask,
                                              nomask=self.nomask)
        flo = conf * flo + (1 - conf) * (flow_attn @ flo)
        flo_0 = flo.reshape(batch, ht, wd, 2).permute(0, 3, 1, 2).contiguous()
        init_flows = [flo_match, flo_0]
        if self.multi_scale:
            flo_s = []
            for ii in range(self.level_corr):
                # corr_i = corr_pyramid[ii + 1]
                corr_i = F.avg_pool2d(corr_i, 2, stride=2)
                h_d, w_d = corr_i.shape[-2:]
                corr_sm = torch.softmax(corr_i.view(batch, ht * wd, h_d * w_d), dim=-1)

                crds_d = coords_grid(batch, h_d, w_d, device=corr_sm.device).view(batch, 2, h_d * w_d).permute(0, 2, 1)
                # flo = torch.einsum('b s m, b m f -> b s f', corr_sm, crds_d) * (ht / h_d) - crds
                crds_d = crds_d * torch.tensor([wd / w_d, ht / h_d]).cuda().view(1, 1, 2)
                flo = torch.einsum('b s m, b m f -> b s f', corr_sm, crds_d) - crds

                flow_attn, conf, _ = self.cfp(self_corr=self_corr, corr_sm=corr_sm, thres=0.4 * 0.8 ** (ii + 1))

                flo = conf * flo + (1 - conf) * (flow_attn @ flo)
                flo = flo.view(batch, ht, wd, 2).permute(0, 3, 1, 2).contiguous()

                flo_s.append(flo)
                init_flows.append(flo)
            flos = torch.cat(flo_s, dim=1)

            flo_s = []
            for conv in self.conv_s:
                flo = conv(flos)
                flo_s.append(flo)
            flos = torch.cat(flo_s, dim=1)
            flo = self.conv_rd(flos)
            init_flows.append(flo)
            flo_0 = flo_0 + self.gamma * flo # TODO learn a weight mask

        return flo_0, init_flows


class FlowEncoder(nn.Module):
    def __init__(self, in_dim=2, out_dim=32):
        super().__init__()
        # 提取光流的高阶特征（方向、幅度、曲率等）
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_dim, 3, padding=1)
        )
        
    def forward(self, flow):
        flow_feat = self.conv(flow)  # [B, 32, H, W]
        return flow_feat


class ContextFusion(nn.Module):
    def __init__(self, flow_dim=32, ctx_dim=128, hidden_dim=128):
        super().__init__()
        # 联合编码图像上下文与光流特征
        self.fusion = nn.Sequential(
            nn.Conv2d(flow_dim + ctx_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, flow_dim, 3, padding=1)
        )
        
    def forward(self, flow_feat, ctx_feat):
        # ctx_feat需下采样/flow_feat需上采样至相同分辨率
        fused = torch.cat([flow_feat, ctx_feat], dim=1)
        return self.fusion(fused)


class DynamicWeightPredictor(nn.Module):
    def __init__(self, in_dim=128, num_flows=4):
        super().__init__()
        # 预测空间自适应权重（每个像素对各光流的置信度）
        self.weight_net = nn.Sequential(
            nn.Conv2d(in_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_flows, 3, padding=1),
            nn.Softmax(dim=1)  # 权重归一化
        )
        
    def forward(self, fused_feat):
        return self.weight_net(fused_feat)  # [B, 4, H, W]


class FlowRefiner(nn.Module):
    def __init__(self, in_dim=2, ctx_dim=128):
        super().__init__()
        # 基于上下文特征的光流残差修正
        self.refine = nn.Sequential(
            nn.Conv2d(in_dim + ctx_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, padding=1)
        )
        
    def forward(self, fused_flow, ctx_feat):
        residual = self.refine(torch.cat([fused_flow, ctx_feat], dim=1))
        return fused_flow + residual  # [B, 2, H, W]


class MultiFlowFusion(nn.Module):
    def __init__(self, ctx_dim=128):
        super().__init__()
        # 初始化各组件
        self.level_corr = 4
        self.flow_encoders = nn.ModuleList([FlowEncoder() for _ in range(4)])
        self.context_fusers = nn.ModuleList([ContextFusion(ctx_dim=ctx_dim) for _ in range(4)])
        self.weight_predictor = DynamicWeightPredictor()
        self.refiner = FlowRefiner(ctx_dim=ctx_dim)
        
    def forward(self, corr_pyramid, ctx_feat):
        """ 
        输入: 
            flows: List of 4光流 [B,2,H,W] 
            ctx_feat: 图像上下文特征 [B,C,H,W]
        输出: 
            refined_flow: 融合优化后的光流 [B,2,H,W]
        """
        batch, _, ht, wd = ctx_feat.shape
        crds = coords_grid(batch, ht, wd, device=ctx_feat.device).view(batch, 2, ht*wd).permute(0, 2, 1)
        corr_i = corr_pyramid[0]
        corr_sm = torch.softmax(corr_i.reshape(batch, ht * wd, ht * wd), dim=-1)
        flo = (corr_sm @ crds) - crds
        flo = flo.reshape(batch, ht, wd, 2).permute(0, 3, 1, 2)
        flows = [flo]
        for ii in range(1, self.level_corr):
            corr_i = corr_pyramid[ii]
            h_d, w_d = corr_i.shape[-2:]
            corr_sm = torch.softmax(corr_i.view(batch, ht * wd, h_d * w_d), dim=-1)

            crds_d = coords_grid(batch, h_d, w_d, device=corr_sm.device)
            # crds_d = coords_grid(batch, h_d, w_d, device=corr_sm.device) + 0.5 # align with 1/8 resolution
            # if ii == 3:
            #     print(ctx_feat.shape)
            #     print(corr_i.shape)
            #     print(crds_d[0])
            #     print((crds_d * (ht / h_d))[0])
            #     print((crds_d * torch.tensor([wd / w_d, ht / h_d]).cuda().view(1, 2, 1, 1))[0])
            crds_d = crds_d * torch.tensor([wd / w_d, ht / h_d]).cuda().view(1, 2, 1, 1)
            crds_d = crds_d.view(batch, 2, -1).permute(0, 2, 1)
            flo = (corr_sm @ crds_d) - crds
            flo = flo.reshape(batch, ht, wd, 2).permute(0, 3, 1, 2)
            flows.append(flo)

        # Step 1: 光流特征提取
        flow_feats = [enc(flow) for enc, flow in zip(self.flow_encoders, flows)]
        # Step 2: 上下文-光流特征融合
        fused_feats = [fuser(feat, ctx_feat) for fuser, feat in zip(self.context_fusers, flow_feats)]
        # Step 3: 动态权重预测
        weights = self.weight_predictor(torch.cat(fused_feats, dim=1))  # 平均融合
        # Step 4: 加权融合
        weighted_flows = torch.stack([w[:, None] * flow for w, flow in zip(weights.unbind(dim=1), flows)], dim=1)
        fused_flow = weighted_flows.sum(dim=1)  # [B,2,H,W]
        flows.append(fused_flow)

        # Step 5: 残差细化
        fused_flow = self.refiner(fused_flow, ctx_feat)
        return fused_flow, flows