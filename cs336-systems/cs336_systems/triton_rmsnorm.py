import torch
from torch import nn
import torch.autograd as autograd
import triton
import triton.language as tl


class RMSNormPyTorch(autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        """
        Args:
            x: tensor where the last dimension has shape H
            weight: vector of dimension H
        """
        # Store for backward pass
        ctx.save_for_backward(x, weight)
        
        dim = x.size(-1)
        mean_square = torch.mean(x.pow(2), dim=-1, keepdim=True)
        rsqrt_norm = torch.rsqrt(mean_square + 1e-5)
        
        output = x * rsqrt_norm * weight
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = 1e-5

        grad_x = rmsnorm_backward_x(grad_output, x, weight, eps)
        grad_g = rmsnorm_backward_g(grad_output, x, weight, eps)

        return grad_x, grad_g


class RMSNormTriton(autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        """
        Args:
            x: tensor where the last dimension has shape H
            weight: vector of dimension H
        """
        # Store for backward pass
        ctx.save_for_backward(x, weight)
        
        input_shape = x.shape
        n_rows = input_shape[0]
        for i in range(1, len(input_shape) - 1):
            n_rows *= input_shape[i]
        n_cols = input_shape[-1]
        
        x_2d = x.view(-1, n_cols)
        
        output = torch.empty_like(x_2d)
        
        stride_x_row = x_2d.stride(0)
        stride_x_col = x_2d.stride(1)
        stride_out_row = output.stride(0)
        stride_out_col = output.stride(1)
        
        BLOCK_SIZE = 128
        
        grid = (n_rows,)
        rmsnorm_forward_kernel[grid](
            x_2d.contiguous(), weight, output,
            n_cols, n_rows,
            stride_x_row, stride_x_col,
            stride_out_row, stride_out_col,
            1e-5,  # epsilon
            BLOCK_SIZE,
        )
        
        output = output.view(input_shape)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        input_shape = x.shape
        n_rows = x.view(-1, x.shape[-1]).shape[0]
        n_cols = x.shape[-1]
        BLOCK_SIZE = 128

        x_2d = x.view(-1, n_cols).contiguous()
        grad_output_2d = grad_output.view(-1, n_cols).contiguous()

        dx = torch.empty_like(x_2d)
        dweight = torch.zeros_like(weight)

        grid = (n_rows,)
        rmsnorm_backward_kernel[grid](
            x_2d, weight, grad_output_2d,
            dx, dweight,
            n_cols, n_rows,
            x_2d.stride(0), x_2d.stride(1),
            grad_output_2d.stride(0), grad_output_2d.stride(1),
            dx.stride(0), dx.stride(1),
            1e-5, BLOCK_SIZE
        )

        return dx.view(input_shape), dweight


class RMSNormTritonModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        return RMSNormTriton.apply(x, self.weight)


class RMSNormPyTorchModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        return RMSNormPyTorch.apply(x, self.weight)


@triton.jit
def rmsnorm_forward_kernel(
    x_ptr,         
    weight_ptr,    
    output_ptr,    
    n_cols,        
    n_rows,        
    stride_x_row,  
    stride_x_col,  
    stride_out_row,
    stride_out_col,
    eps,           
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    if row_idx < n_rows:
        x_row_ptr = x_ptr + row_idx * stride_x_row
        output_row_ptr = output_ptr + row_idx * stride_out_row
        
        sum_squares = 0.0
        for col_start in range(0, n_cols, BLOCK_SIZE):
            col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < n_cols
            
            x_vals = tl.load(x_row_ptr + col_offsets * stride_x_col, mask=mask, other=0.0)
            
            sum_squares += tl.sum(x_vals * x_vals * mask)
        
        mean_square = sum_squares / n_cols
        rsqrt_norm = 1.0 / tl.sqrt(mean_square + eps)
        
        for col_start in range(0, n_cols, BLOCK_SIZE):
            col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < n_cols
            
            x_vals = tl.load(x_row_ptr + col_offsets * stride_x_col, mask=mask, other=0.0)
            weight_vals = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
            
            output_vals = x_vals * rsqrt_norm * weight_vals
            tl.store(output_row_ptr + col_offsets * stride_out_col, output_vals, mask=mask)


@triton.jit
def rmsnorm_backward_kernel(
    x_ptr, weight_ptr, grad_out_ptr, dx_ptr, dweight_ptr,
    n_cols, n_rows,
    stride_x_row, stride_x_col,
    stride_gout_row, stride_gout_col,
    stride_dx_row, stride_dx_col,
    eps, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    if row >= n_rows:
        return

    x_row_ptr = x_ptr + row * stride_x_row
    gout_row_ptr = grad_out_ptr + row * stride_gout_row
    dx_row_ptr = dx_ptr + row * stride_dx_row

    sum_sq = 0.0
    for col in range(0, n_cols, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x_vals = tl.load(x_row_ptr + offs * stride_x_col, mask=mask, other=0.0)
        sum_sq += tl.sum(x_vals * x_vals * mask)

    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    rms3 = rms * mean_sq

    dot = 0.0
    for col in range(0, n_cols, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols

        x_vals = tl.load(x_row_ptr + offs * stride_x_col, mask=mask, other=0.0)
        gout_vals = tl.load(gout_row_ptr + offs * stride_gout_col, mask=mask, other=0.0)
        weight_vals = tl.load(weight_ptr + offs, mask=mask, other=0.0)

        gz = gout_vals * weight_vals
        dot += tl.sum(x_vals * gz * mask)

    for col in range(0, n_cols, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols

        x_vals = tl.load(x_row_ptr + offs * stride_x_col, mask=mask, other=0.0)
        gout_vals = tl.load(gout_row_ptr + offs * stride_gout_col, mask=mask, other=0.0)
        weight_vals = tl.load(weight_ptr + offs, mask=mask, other=0.0)

        gz = gout_vals * weight_vals
        dx_vals = (gz / rms - x_vals * dot / (n_cols * rms3))
        tl.store(dx_row_ptr + offs * stride_dx_col, dx_vals, mask=mask)

        dweight_add = gout_vals * x_vals / rms
        tl.atomic_add(dweight_ptr + offs, dweight_add, mask=mask)


def rmsnorm_backward_g(grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Computes ∇_g L for RMSNorm.

    Parameters:
    - grad_output: Tensor of shape (..., H), gradient of the loss with respect to RMSNorm output.
    - x: Tensor of shape (..., H), the input tensor to RMSNorm.
    - g: Tensor of shape (H,), the scale parameter.
    - eps: Small epsilon for numerical stability.

    Returns:
    - grad_g: Tensor of shape (H,), the gradient with respect to g.
    """
    # Compute the RMS denominator per row
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)  # shape (..., 1)
    
    # Normalize x
    z = x / rms  # shape (..., H)

    # Compute ∇_g L = sum over all dimensions except last of grad_output * z
    grad_g = torch.sum(grad_output * z, dim=tuple(range(grad_output.ndim - 1)))  # shape (H,)
    return grad_g

def rmsnorm_backward_x(grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Computes ∇_x L for RMSNorm.

    Parameters:
    - grad_output: Tensor of shape (..., H), gradient of the loss with respect to RMSNorm output.
    - x: Tensor of shape (..., H), input to RMSNorm.
    - g: Tensor of shape (H,), scale parameter.
    - eps: Epsilon for numerical stability.

    Returns:
    - grad_x: Tensor of shape (..., H), gradient with respect to x.
    """
    H = x.shape[-1]
    
    # Compute rms
    rms_sq = x.pow(2).mean(dim=-1, keepdim=True) + eps
    rms = torch.sqrt(rms_sq)
    rms_cubed = rms_sq * rms

    # z = x / rms, dy/dx = g * d(x / rms) / dx
    grad_output_scaled = grad_output * g  # (..., H)

    # Dot(x, grad_output_scaled)
    dot = torch.sum(x * grad_output_scaled, dim=-1, keepdim=True)

    # ∇xL = grad_output_scaled / rms - (x * dot) / (H * rms^3)
    grad_x = grad_output_scaled / rms - (x * dot) / (H * rms_cubed)
    return grad_x
