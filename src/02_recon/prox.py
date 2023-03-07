import numpy as np
import sigpy as sp

class L1Wav(sp.prox.Prox):
  def __init__(self, shape, lamda, axes=None):
    self.lamda = lamda
    self.W = sp.linop.Wavelet(shape, axes=axes)
    super().__init__(shape)

  def _prox(self, alpha, input):
    dev = sp.get_device(input)
    xp = dev.xp
    with dev:
      input_shape = input.shape

      phase = xp.exp(1j * xp.random.uniform(low=-xp.pi, high=xp.pi))

      input *= phase
      input = self.W.H(sp.thresh.soft_thresh(self.lamda * alpha,
                                             self.W(input)))
      input *= xp.conj(phase)
      return input

def _llr(x, lamda, N, L, w, shift):

  device = sp.get_device(x)
  xp = device.xp
  with device:
    for k in range(N):
      x = xp.roll(x, shift, axis=-(k + 1))
    mats = L(x)
    (u, s, vh) = xp.linalg.svd(mats, full_matrices=False)
    thresh_s = s - lamda
    thresh_s[thresh_s < 0] = 0
    mats[...] = xp.matmul(u * thresh_s[..., None, :], vh)
    x = L.H(mats)
    if w is not None:
      x = x/w[None, ...]
    for k in range(N):
      x = xp.roll(x, -shift, axis=-(k + 1))
    return x

class LLR(sp.prox.Prox):
  def __init__(self, shape, lamda, block, devnum, stride=None):
    self.N = len(shape[1:])
    assert self.N == 2 or self.N == 3

    self.lamda = lamda
    self.block = block

    if stride is None:
      stride = block

    B = sp.linop.ArrayToBlocks(shape[1:], (block,)*self.N, (stride,)*self.N)

    if stride != block:
      self.w = (B.H * B)(np.ones(B.ishape, dtype=np.complex64))
    else:
      self.w = None

    B = sp.linop.ArrayToBlocks(shape, (block,)*self.N, (stride,)*self.N)
    if self.N == 3:
      T = sp.linop.Transpose(B.oshape, (1, 2, 3, 0, 4, 5, 6))
      n = T.oshape[0] * T.oshape[1] * T.oshape[2]
    else:
      T = sp.linop.Transpose(B.oshape, (1, 2, 0, 3, 4))
      n = T.oshape[0] * T.oshape[1]
    R = sp.linop.Reshape((n, shape[0], block**self.N), T.oshape)
    self.L = R * T * B

    self.c = 0
    super().__init__(shape)

  def _prox(self, alpha, input):
    self.c = self.c + 1
    if self.w is not None:
      self.w = sp.to_device(self.w, sp.get_device(input))
    return _llr(input, self.lamda * alpha, self.N, self.L, self.w, self.c)
