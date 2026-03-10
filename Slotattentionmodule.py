import torch
import torch.nn as nn


class WeightedAttention(nn.Module):
    def __init__(self, dim, eps = 1e-8, softmax_dim = 1, weighted_mean_dim = 2):
        super().__init__()
        self.norm_input = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.eps = eps
        self.scale = dim ** -0.5
        self.softmax_dim = softmax_dim
        self.weighted_mean_dim = weighted_mean_dim

    def forward(self, inputs, context):

        inputs = self.norm_input(inputs)
        context = self.norm_context(context)

        q = self.to_q(inputs)
        k = self.to_k(context)
        v = self.to_v(context)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim = self.softmax_dim) + self.eps
        attn = attn / attn.sum(dim = self.weighted_mean_dim, keepdim=True)

        updates = torch.einsum('bjd,bij->bid', v, attn)
        return updates

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

class GatedResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.fn = fn
    def forward(self, *args):
        inputs = args[0]
        b, _, d = inputs.shape

        updates = self.fn(*args)

        inputs = self.gru(
            updates.reshape(-1, d),
            inputs.reshape(-1, d)
        )
        return inputs.reshape(b, -1, d)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_dim = max(dim, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)


class MLPs(nn.Module):

    def __init__(self, fea_dim=20,
                 out_pt_fea_dim=64):
        super(MLPs, self).__init__()

        self.PPmodel = nn.Sequential(
            # nn.BatchNorm1d(fea_dim),

            nn.Linear(fea_dim, out_pt_fea_dim),
            # nn.BatchNorm1d(64),
            nn.LayerNorm(out_pt_fea_dim),
            nn.ReLU(),

            # nn.Linear(64, out_pt_fea_dim),
            # # nn.BatchNorm1d(128),
            # nn.LayerNorm(out_pt_fea_dim),
            # nn.ReLU()

        )

    def forward(self, x):
        x = self.PPmodel(x)
        return x



class SlotAttention(nn.Module):
    """Slot Attention module."""

    def __init__(self, num_slots, encoder_dims, iters=3, hidden_dim=128, eps=1e-8):
        """Builds the Slot Attention module.
        Args:
            iters: Number of iterations.
            num_slots: Number of slots.
            encoder_dims: Dimensionality of slot feature vectors.
            hidden_dim: Hidden layer size of MLP.
            eps: Offset for attention coefficients before normalization.
        """
        super(SlotAttention, self).__init__()

        self.eps = eps
        self.iters = iters
        self.num_slots = num_slots
        self.scale = encoder_dims ** -0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.norm_input = nn.LayerNorm(encoder_dims)
        self.norm_slots = nn.LayerNorm(encoder_dims)
        self.norm_pre_ff = nn.LayerNorm(encoder_dims)

        # Parameters for Gaussian init (shared by all slots).
        # self.slots_mu = nn.Parameter(torch.randn(1, 1, encoder_dims))
        # self.slots_sigma = nn.Parameter(torch.randn(1, 1, encoder_dims))

        self.slots_embedding = nn.Embedding(num_slots, encoder_dims)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(encoder_dims, encoder_dims)
        self.project_k = nn.Linear(encoder_dims, encoder_dims)
        self.project_v = nn.Linear(encoder_dims, encoder_dims)

        # Slot update functions.
        self.gru = nn.GRUCell(encoder_dims, encoder_dims)

        hidden_dim = max(encoder_dims, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dims, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, encoder_dims)
        )

    def forward(self, inputs, num_slots=None):
        # inputs has shape [batch_size, num_inputs, inputs_size].
        inputs = self.norm_input(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        # random slots initialization,
        # mu = self.slots_mu.expand(b, n_s, -1)
        # sigma = self.slots_sigma.expand(b, n_s, -1)
        # slots = torch.normal(mu, sigma)

        # learnable slots initialization
        slots = self.slots_embedding(torch.arange(0, n_s).expand(b, n_s).to(self.device))

        # Multiple rounds of attention.
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean.

            updates = torch.einsum('bjd,bij->bid', v, attn)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


def truncated_normal_(tensor, mean=0, std=.02):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4, )).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor
class SoftPositionEmbed(nn.Module):
    """Adds soft positional embedding with learnable projection."""

    def __init__(self, hidden_size):
        """Builds the soft position embedding layer.
        Args:
          hidden_size: Size of input feature dimension.
          resolution: Tuple of integers specifying width and height of grid.
        """
        super(SoftPositionEmbed, self).__init__()
        self.proj = nn.Linear(4, hidden_size)
        # self.grid = build_grid(resolution)
        self.pos_embedd = nn.Parameter(truncated_normal_(torch.randn(1, 1, hidden_size)))

    def forward(self, inputs):
        # b,x,c
        return inputs + self.pos_embedd # + self.proj(self.grid)

class FeedForward1(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class SlotAttentionExperimental(nn.Module):
    def __init__(self, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()

        self.iters = iters

        self.norm_inputs = nn.LayerNorm(dim)



        self.slots_to_inputs_attn = GatedResidual(dim, WeightedAttention(dim, eps = eps))
        self.slots_ff = GatedResidual(dim, FeedForward(dim, hidden_dim))

        self.inputs_to_slots_attn = GatedResidual(dim, WeightedAttention(dim, eps = eps, softmax_dim = 2, weighted_mean_dim = 1))
        self.inputs_ff = GatedResidual(dim, FeedForward(dim, hidden_dim))

    def forward(self, inputs, slots = None):


        inputs = self.norm_inputs(inputs)

        for _ in range(self.iters):
            slots = self.slots_to_inputs_attn(slots, inputs)
            slots = self.slots_ff(slots)

            inputs = self.inputs_to_slots_attn(inputs, slots)
            inputs = self.inputs_ff(inputs)

        return slots, inputs


class SlotAttentionAutoEncoder(nn.Module):
        """Slot Attention-based auto-encoder for object discovery."""

        def __init__(self, num_slots, iters=5):
            """Builds the Slot Attention-based Auto-encoder.
            Args:
                resolution: Tuple of integers specifying width and height of input image
                num_slots: Number of slots in Slot Attention.
                iters: Number of iterations in Slot Attention.
            """
            super(SlotAttentionAutoEncoder, self).__init__()

            self.iters = iters
            self.num_slots = num_slots
            self.encoder_arch = [64, 'MP', 64, 'MP', 64]
            self.encoder_dims = self.encoder_arch[-1]

            self.posemd = SoftPositionEmbed(64)

            self.layer_norm = nn.LayerNorm(self.encoder_dims)

            self.mlp = nn.Sequential(
                nn.Linear(self.encoder_dims, self.encoder_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.encoder_dims, self.encoder_dims)
            )

            self.norm_inputs1 = nn.LayerNorm(self.encoder_dims)

            self.slot_attention1 = SlotAttention(
                iters=self.iters,
                num_slots=self.num_slots,
                encoder_dims=self.encoder_dims,
                hidden_dim=self.encoder_dims)


            self.mlp1 = FeedForward1(3, 64)
            self.project_q = nn.Linear(64, 1)

            # self.slot_attention5 = SlotAttention(
            #     iters=self.iters,
            #     num_slots=self.num_slots,
            #     encoder_dims=self.encoder_dims,
            #     hidden_dim=self.encoder_dims)

        def forward(self, input):


            x =self.posemd(input)
            # print( x.shape, "x")
            x = self.mlp(self.layer_norm(x))  # Feedforward network on set.
            slots1 = self.slot_attention1(x)  # torch.Size([96, 20, 128]) slots
            # slots1 = torch.squeeze(self.project_q(slots1))


            return slots1



        #####################OLD slot attention module #################################

        # class SlotAttention(nn.Module):
        #     """Slot Attention module."""
        #
        #     def __init__(self, num_slots, encoder_dims, iters=3, hidden_dim=128, eps=1e-8):
        #         """Builds the Slot Attention module.
        #         Args:
        #             iters: Number of iterations.
        #             num_slots: Number of slots.
        #             encoder_dims: Dimensionality of slot feature vectors.
        #             hidden_dim: Hidden layer size of MLP.
        #             eps: Offset for attention coefficients before normalization.
        #         """
        #         super(SlotAttention, self).__init__()
        #
        #         self.eps = eps
        #         self.iters = iters
        #         self.num_slots = num_slots
        #         self.scale = encoder_dims ** -0.5
        #         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #
        #         self.norm_input = nn.LayerNorm(encoder_dims)
        #         self.norm_slots = nn.LayerNorm(encoder_dims)
        #         self.norm_pre_ff = nn.LayerNorm(encoder_dims)
        #
        #         # Parameters for Gaussian init (shared by all slots).
        #         # self.slots_mu = nn.Parameter(torch.randn(1, 1, encoder_dims))
        #         # self.slots_sigma = nn.Parameter(torch.randn(1, 1, encoder_dims))
        #
        #         self.slots_embedding = nn.Embedding(num_slots, encoder_dims)
        #
        #         # Linear maps for the attention module.
        #         self.project_q = nn.Linear(encoder_dims, encoder_dims)
        #         self.project_k = nn.Linear(encoder_dims, encoder_dims)
        #         self.project_v = nn.Linear(encoder_dims, encoder_dims)
        #
        #         # Slot update functions.
        #         self.gru = nn.GRUCell(encoder_dims, encoder_dims)
        #
        #         hidden_dim = max(encoder_dims, hidden_dim)
        #         self.mlp = nn.Sequential(
        #             nn.Linear(encoder_dims, hidden_dim),
        #             nn.ReLU(inplace=True),
        #             nn.Linear(hidden_dim, encoder_dims)
        #         )
        #
        #     def forward(self, inputs, num_slots=None):
        #         # inputs has shape [batch_size, num_inputs, inputs_size].
        #         inputs = self.norm_input(inputs)  # Apply layer norm to the input.
        #         k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        #         v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        #
        #         # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        #         b, n, d = inputs.shape
        #         n_s = num_slots if num_slots is not None else self.num_slots
        #
        #         # random slots initialization,
        #         # mu = self.slots_mu.expand(b, n_s, -1)
        #         # sigma = self.slots_sigma.expand(b, n_s, -1)
        #         # slots = torch.normal(mu, sigma)
        #
        #         # learnable slots initialization
        #         slots = self.slots_embedding(torch.arange(0, n_s).expand(b, n_s).to(self.device))
        #
        #         # Multiple rounds of attention.
        #         for _ in range(self.iters):
        #             slots_prev = slots
        #             slots = self.norm_slots(slots)
        #
        #             # Attention.
        #             q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
        #             dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        #             attn = dots.softmax(dim=1) + self.eps
        #             attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean.
        #
        #             updates = torch.einsum('bjd,bij->bid', v, attn)
        #             # `updates` has shape: [batch_size, num_slots, slot_size].
        #
        #             # Slot update.
        #             slots = self.gru(
        #                 updates.reshape(-1, d),
        #                 slots_prev.reshape(-1, d)
        #             )
        #             slots = slots.reshape(b, -1, d)
        #             slots = slots + self.mlp(self.norm_pre_ff(slots))
        #
        #         return slots
        #
        # def truncated_normal_(tensor, mean=0, std=.02):
        #     size = tensor.shape
        #     tmp = tensor.new_empty(size + (4,)).normal_()
        #     valid = (tmp < 2) & (tmp > -2)
        #     ind = valid.max(-1, keepdim=True)[1]
        #     tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        #     tensor.data.mul_(std).add_(mean)
        #     return tensor
        #
        # class SoftPositionEmbed(nn.Module):
        #     """Adds soft positional embedding with learnable projection."""
        #
        #     def __init__(self, hidden_size):
        #         """Builds the soft position embedding layer.
        #         Args:
        #           hidden_size: Size of input feature dimension.
        #           resolution: Tuple of integers specifying width and height of grid.
        #         """
        #         super(SoftPositionEmbed, self).__init__()
        #         self.proj = nn.Linear(4, hidden_size)
        #         # self.grid = build_grid(resolution)
        #         self.pos_embedd = nn.Parameter(truncated_normal_(torch.randn(1, 1, hidden_size)))
        #
        #     def forward(self, inputs):
        #         # b,x,c
        #         return inputs + self.pos_embedd  # + self.proj(self.grid)
        #
        # class FeedForward(nn.Module):
        #     def __init__(self, dim, hidden_dim, dropout=0.):
        #         super().__init__()
        #         self.net = nn.Sequential(
        #             nn.Linear(dim, hidden_dim),
        #             nn.GELU(),
        #             nn.Dropout(dropout),
        #             nn.Linear(hidden_dim, hidden_dim),
        #             nn.Dropout(dropout)
        #         )
        #
        #     def forward(self, x):
        #         return self.net(x)
        #
        # class SlotAttentionAutoEncoder(nn.Module):
        #     """Slot Attention-based auto-encoder for object discovery."""
        #
        #     def __init__(self, num_slots, iters=5):
        #         """Builds the Slot Attention-based Auto-encoder.
        #         Args:
        #             resolution: Tuple of integers specifying width and height of input image
        #             num_slots: Number of slots in Slot Attention.
        #             iters: Number of iterations in Slot Attention.
        #         """
        #         super(SlotAttentionAutoEncoder, self).__init__()
        #
        #         self.iters = iters
        #         self.num_slots = num_slots
        #         self.encoder_arch = [64, 'MP', 64, 'MP', 64]
        #         self.encoder_dims = self.encoder_arch[-1]
        #         # self.Features = cylinder_fea()
        #
        #         self.posemd = SoftPositionEmbed(64)
        #
        #         self.layer_norm = nn.LayerNorm(self.encoder_dims)
        #
        #         self.mlp = nn.Sequential(
        #             nn.Linear(self.encoder_dims, self.encoder_dims),
        #             nn.ReLU(inplace=True),
        #             nn.Linear(self.encoder_dims, self.encoder_dims)
        #         )
        #
        #         self.slot_attention1 = SlotAttention(
        #             iters=self.iters,
        #             num_slots=self.num_slots,
        #             encoder_dims=self.encoder_dims,
        #             hidden_dim=self.encoder_dims)
        #
        #         self.slot_attention2 = SlotAttention(
        #             iters=self.iters,
        #             num_slots=self.num_slots,
        #             encoder_dims=self.encoder_dims,
        #             hidden_dim=self.encoder_dims)
        #
        #         self.slot_attention3 = SlotAttention(
        #             iters=self.iters,
        #             num_slots=self.num_slots,
        #             encoder_dims=self.encoder_dims,
        #             hidden_dim=self.encoder_dims)
        #
        #         self.slot_attention4 = SlotAttention(
        #             iters=self.iters,
        #             num_slots=self.num_slots,
        #             encoder_dims=self.encoder_dims,
        #             hidden_dim=self.encoder_dims)
        #
        #         self.slot_attention5 = SlotAttention(
        #             iters=self.iters,
        #             num_slots=self.num_slots,
        #             encoder_dims=self.encoder_dims,
        #             hidden_dim=self.encoder_dims)
        #
        #         self.mlp1 = FeedForward(20, 64)
        #         self.mlp2 = FeedForward(20, 64)
        #         self.mlp3 = FeedForward(20, 64)
        #         self.mlp4 = FeedForward(20, 64)
        #         self.mlp5 = FeedForward(20, 64)
        #
        #     def forward(self, input):
        #         x = self.posemd(input)
        #
        #         x = self.mlp(self.layer_norm(x))  # Feedforward network on set.
        #
        #         slots1 = self.slot_attention1(x)  # torch.Size([96, 20, 128]) slots
        #         x1 = torch.bmm(x, slots1.transpose(-1, -2))
        #         x1 = self.mlp1(x1)
        #
        #         slots2 = self.slot_attention2(x1)  # torch.Size([96, 20, 128]) slots
        #         x2 = torch.bmm(x1, slots2.transpose(-1, -2))
        #         x2 = self.mlp2(x2)
        #
        #         slots3 = self.slot_attention3(x2)  # torch.Size([96, 20, 128]) slots
        #         x3 = torch.bmm(x2, slots3.transpose(-1, -2))
        #         x3 = self.mlp3(x3)
        #         x3 = x3 + x2
        #
        #         slots4 = self.slot_attention4(x3)  # torch.Size([96, 20, 128]) slots
        #         x4 = torch.bmm(x3, slots4.transpose(-1, -2))
        #         x4 = self.mlp4(x4)
        #         x4 = x4 + x1
        #
        #         slots5 = self.slot_attention5(x4)  # torch.Size([96, 20, 128]) slots
        #         x5 = torch.bmm(x4, slots5.transpose(-1, -2))
        #
        #         x5 = self.mlp5(x5)
        #
        #         return x5
        #


