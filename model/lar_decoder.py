import torch
import torch.nn as nn
import torch.nn.functional as F

class LAR_Net_Decoder_V27(nn.Module):
    def __init__(self, fpn_in_dims, fpn_out_dim, word_dim, state_dim, n_iter=2):
        super().__init__()
        
        SepConv = lambda in_c, out_c, k_size: DepthwiseSeparableConv(in_c, out_c, kernel_size=k_size, padding=k_size//2)
        
        self.shared_injector = GlobalContext_Injector(fpn_out_dim, fpn_out_dim, use_separable_conv=True)

        self.bottleneck_conv = SepConv(fpn_in_dims[2], fpn_out_dim, 3)
        self.cell_neck = Heavy_FusionRefine_Cell(fpn_out_dim, fpn_out_dim, word_dim, state_dim, n_iter, use_separable_conv=True)
        
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # align_corners=False
                                 SepConv(fpn_out_dim, fpn_out_dim, 3),
                                 nn.GroupNorm(fpn_out_dim//8, fpn_out_dim), nn.ReLU(inplace=True))
        

        self.fuse1_proj = SepConv(fpn_out_dim + fpn_in_dims[1], fpn_out_dim, 1) 
        self.fuse1_resblock = ResidualConvBlock(fpn_out_dim, use_separable_conv=True)
        
        self.cell1 = Medium_FusionRefine_Cell(fpn_out_dim, fpn_out_dim, word_dim, state_dim, n_iter, use_separable_conv=True)
        
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # align_corners=False
                                 SepConv(fpn_out_dim, fpn_out_dim, 3),
                                 nn.GroupNorm(fpn_out_dim//8, fpn_out_dim), nn.ReLU(inplace=True))
                                 

        self.fuse2_proj = SepConv(fpn_out_dim + fpn_in_dims[0], fpn_out_dim, 1) 
        self.fuse2_resblock = ResidualConvBlock(fpn_out_dim, use_separable_conv=True) 
        
        self.cell2 = Light_FusionRefine_Cell(fpn_out_dim, fpn_out_dim, word_dim, state_dim, n_iter, use_separable_conv=True)
        
        final_up_dim = fpn_out_dim // 2
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False), # align_corners=False
            SepConv(fpn_out_dim, final_up_dim, 3),
            nn.GroupNorm(max(1, final_up_dim // 8), final_up_dim),
            nn.ReLU(inplace=True),
            ResidualConvBlock(final_up_dim, use_separable_conv=True) 
        )

        self.seg_head_neck = nn.Conv2d(fpn_out_dim, 1, kernel_size=1)
        self.seg_head1 = nn.Conv2d(fpn_out_dim, 1, kernel_size=1)
        self.main_seg_head = nn.Conv2d(final_up_dim, 1, kernel_size=1)

    def forward(self, vis_outs, fq_context, word, state):
        v_skip0, v_skip1, v_skip2 = vis_outs
        
        v_neck = self.bottleneck_conv(v_skip2)
        v_neck_enhanced = self.shared_injector(v_neck, fq_context)
        v_out_neck = self.cell_neck(v_neck_enhanced, word, state)
        logit_neck = self.seg_head_neck(v_out_neck)
        
        v_in1 = self.up1(v_out_neck)
        fused1_cat = torch.cat([v_in1, v_skip1], dim=1)
        fused1_proj = self.fuse1_proj(fused1_cat)
        v_fused1 = self.fuse1_resblock(fused1_proj)
        
        v_fused1_enhanced = self.shared_injector(v_fused1, fq_context)
        v_out1 = self.cell1(v_fused1_enhanced, word, state)
        logit1 = self.seg_head1(v_out1)
        
        v_in2 = self.up2(v_out1)
        fused2_cat = torch.cat([v_in2, v_skip0], dim=1)
        fused2_proj = self.fuse2_proj(fused2_cat)
        v_fused2 = self.fuse2_resblock(fused2_proj)
        
        v_fused2_enhanced = self.shared_injector(v_fused2, fq_context)
        v_out2 = self.cell2(v_fused2_enhanced, word, state)
        
        # Final stage (now deeper)
        v_final = self.final_up(v_out2)
        main_logit = self.main_seg_head(v_final)
        
        return main_logit, [logit_neck, logit1]
