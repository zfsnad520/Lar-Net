import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, use_separable_conv=False):
        super().__init__()
        self.padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        ConvLayer = DepthwiseSeparableConv if use_separable_conv else nn.Conv2d

        self.conv_gates = ConvLayer(self.input_dim + self.hidden_dim, 2 * self.hidden_dim, kernel_size=kernel_size, padding=self.padding, bias=True)
        self.conv_can = ConvLayer(self.input_dim + self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, padding=self.padding, bias=True)

    def forward(self, input_tensor, h_cur):
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)
        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        h_next = torch.tanh(cc_cnm)
        h_new = (1 - update_gate) * h_cur + update_gate * h_next
        return h_new

class Heavy_FusionRefine_Cell(nn.Module):
    def __init__(self, in_dim, hidden_dim, word_dim, state_dim, n_iter=2, use_separable_conv=False):
        super().__init__()
        self.n_iter = n_iter
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*2, batch_first=True, dropout=0.1)
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.global_gate = nn.Linear(state_dim, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*2, batch_first=True, dropout=0.1)
        self.cross_attn = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.conv_gru_shared = ConvGRUCell(hidden_dim, hidden_dim, kernel_size=3, use_separable_conv=use_separable_conv)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, v_fused, word_features, state_feature):
        b, c, h, w = v_fused.shape
        v_seq = v_fused.flatten(2).transpose(1, 2)
        v_seq_purified = self.self_attn(v_seq)
        gate = torch.sigmoid(self.global_gate(state_feature)).unsqueeze(1)
        v_seq_gated = v_seq_purified * gate
        v_seq_refined = self.cross_attn(tgt=v_seq_gated, memory=word_features)
        h_current = self.norm(v_seq_refined).transpose(1, 2).reshape(b, c, h, w)
        for _ in range(self.n_iter):
            h_current = self.conv_gru_shared(h_current, h_current)
        return v_fused + h_current

class Medium_FusionRefine_Cell(nn.Module):
    def __init__(self, in_dim, hidden_dim, word_dim, state_dim, n_iter=2, use_separable_conv=False):
        super().__init__()
        self.n_iter = n_iter
        self.medium_cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, kdim=word_dim, vdim=word_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.channel_gate = nn.Linear(state_dim, hidden_dim)
        self.conv_gru_shared = ConvGRUCell(hidden_dim, hidden_dim, kernel_size=3, use_separable_conv=use_separable_conv)

    def forward(self, v_fused, word_features, state_feature):
        b, c, h, w = v_fused.shape
        v_seq = v_fused.flatten(2).transpose(1, 2)
        v_attn, _ = self.medium_cross_attn(query=v_seq, key=word_features, value=word_features)
        v_seq = self.norm(v_seq + v_attn)
        v_refined_by_word = v_seq.transpose(1, 2).reshape(b, c, h, w)
        gate_c = torch.sigmoid(self.channel_gate(state_feature)).unsqueeze(-1).unsqueeze(-1)
        v_final_refined = v_refined_by_word * gate_c
        h_current = v_final_refined
        for _ in range(self.n_iter):
            h_current = self.conv_gru_shared(h_current, h_current)
        return v_fused + h_current

class Light_FusionRefine_Cell(nn.Module):
    def __init__(self, in_dim, hidden_dim, word_dim, state_dim, n_iter=2, use_separable_conv=False):
        super().__init__()
        self.n_iter = n_iter
        self.light_cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, kdim=word_dim, vdim=word_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.channel_gate = nn.Linear(state_dim, hidden_dim)
        self.conv_gru_shared = ConvGRUCell(hidden_dim, hidden_dim, kernel_size=3, use_separable_conv=use_separable_conv)

    def forward(self, v_fused, word_features, state_feature):
        b, c, h, w = v_fused.shape
        v_seq = v_fused.flatten(2).transpose(1, 2)
        v_attn, _ = self.light_cross_attn(query=v_seq, key=word_features, value=word_features)
        v_seq = self.norm(v_seq + v_attn)
        v_refined_by_word = v_seq.transpose(1, 2).reshape(b, c, h, w)
        gate_c = torch.sigmoid(self.channel_gate(state_feature)).unsqueeze(-1).unsqueeze(-1)
        v_final_refined = v_refined_by_word * gate_c
        h_current = v_final_refined
        for _ in range(self.n_iter):
            h_current = self.conv_gru_shared(h_current, h_current)
        return v_fused + h_current

class ResidualConvBlock(nn.Module):
    def __init__(self, dim, use_separable_conv=False):
        super().__init__()
        ConvLayer = DepthwiseSeparableConv if use_separable_conv else nn.Conv2d
        self.conv_block = nn.Sequential(
            ConvLayer(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(dim // 8 if dim >= 16 else 1, dim),
            nn.ReLU(inplace=True),
            ConvLayer(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(dim // 8 if dim >= 16 else 1, dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv_block(x))

class GlobalContext_Injector(nn.Module):
    def __init__(self, in_dim, context_dim, use_separable_conv=False):
        super().__init__()
        self.context_align = nn.Conv2d(context_dim, in_dim, kernel_size=1, bias=False)
        self.gate = nn.Conv2d(in_dim * 2, in_dim, kernel_size=1)
        self.fusion_refiner = ResidualConvBlock(in_dim, use_separable_conv=use_separable_conv)

    def forward(self, local_feature, global_context):
        global_context_aligned = self.context_align(
            F.interpolate(global_context, size=local_feature.shape[-2:], mode='bicubic', align_corners=False)
        )
        combined = torch.cat([local_feature, global_context_aligned], dim=1)
        gate = torch.sigmoid(self.gate(combined))
        fused_feature = local_feature * (1 - gate) + global_context_aligned * gate
        return self.fusion_refiner(fused_feature)

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
