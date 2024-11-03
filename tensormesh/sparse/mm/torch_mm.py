
import torch
from torch.autograd import Function
import scipy.sparse
from ..utils import tensor2cupy, cupy2tensor, shapeT

class SparseMMTorch(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, B):
        C = torch.sparse_coo_tensor(
            torch.stack([row, col]), edata, shape).mm(B)
        ctx.save_for_backward(edata, row, col, B)
        ctx.A_shape = shape
        return C

    @staticmethod
    def backward(ctx, grad_outputs):
        edata, row, col, B = ctx.saved_tensors
        edata_grad  = (grad_outputs[row]* B[col]).sum(dim=1)
        A_T    = torch.sparse_coo_tensor(
            torch.stack([col, row]), edata, shapeT(ctx.A_shape))
        grad_B = A_T.mm(grad_outputs)
        return edata_grad, None, None, None, grad_B
       
class SparseMVTorch(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, B):
        C = torch.sparse_coo_tensor(
            torch.stack([row, col]), edata, shape).mv(B)
        ctx.save_for_backward(edata, row, col, B)
        ctx.A_shape = shape
        return C

    @staticmethod
    def backward(ctx, grad_outputs):
        edata, row, col, B = ctx.saved_tensors  
        edata_grad  = grad_outputs[row]  * B[col]
        A_T    = torch.sparse_coo_tensor(
            torch.stack([col, row]), edata, shapeT(ctx.A_shape))
        grad_B = A_T.mv(grad_outputs)
        return edata_grad, None, None, None, grad_B


