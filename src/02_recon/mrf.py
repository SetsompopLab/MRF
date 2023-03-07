import sigpy as sp
import sigpy.mri as mr

def linop(trj, phi, mps, sqrt_dcf=None, dct_b0=None):

  if dct_b0 == None:
    return base_linop(trj, phi, mps, sqrt_dcf)

  dev = sp.get_device(mps)
  assert sp.get_device(phi) == dev
  xp = dev.xp

  with dev:
    b0_arr = sp.to_device(dct_b0["b0_arr"], dev)

    time = (xp.arange(0, trj.shape[-2]) * dct_b0["dt"]).ravel()
    time_segments = [xp.mean(elm) for elm in
                     xp.array_split(time, dct_b0["num_segments"], axis=0)]
    phase_arrs = [xp.exp(-1j * 2 * xp.pi * t * b0_arr) for t in time_segments]

    lst_trj = xp.array_split(trj, dct_b0["num_segments"], axis=-2)
    if sqrt_dcf is not None:
      lst_dcf = xp.array_split(sqrt_dcf, dct_b0["num_segments"], axis=-1)

    if sqrt_dcf is None:
      lst_A = [base_linop(lst_trj[k], phi, mps, None)
               for k in range(dct_b0["num_segments"])]
    else:
      lst_A = [base_linop(lst_trj[k], phi, mps, lst_dcf[k])
               for k in range(dct_b0["num_segments"])]

    ishape = lst_A[0].ishape
    for k in range(dct_b0["num_segments"]):
      lst_A[k] = lst_A[k] * sp.linop.Multiply(ishape, phase_arrs[k][None, ...])
    A = sp.linop.Vstack(lst_A, axis=3)

  return A

def base_linop(trj, phi, mps, sqrt_dcf=None):

  dev = sp.get_device(mps)
  assert sp.get_device(phi) == dev

  if type(sqrt_dcf) == type(None):
    F = sp.linop.NUFFT(mps.shape[1:], trj)
  else:
    assert sp.get_device(sqrt_dcf) == dev
    F = sp.linop.Multiply(trj.shape[:-1], sqrt_dcf) * \
        sp.linop.NUFFT(mps.shape[1:], trj)

  outer_A = []
  for k in range(mps.shape[0]):
    S = sp.linop.Multiply(mps.shape[1:], mps[k, ...]) * \
        sp.linop.Reshape( mps.shape[1:], [1] + list(mps.shape[1:]))
    lst_A = [sp.linop.Reshape([1] + list(F.oshape), F.oshape)   * \
             sp.linop.Multiply(F.oshape, phi[k, :, None, None]) * \
             F * S for k in range(phi.shape[0])]
    inner_A = sp.linop.Hstack(lst_A, axis=0)
    D1 = sp.linop.ToDevice(inner_A.ishape, dev, sp.cpu_device)
    D2 = sp.linop.ToDevice(inner_A.oshape, sp.cpu_device, dev)
    outer_A.append(D2 * inner_A * D1) 
  A = sp.linop.Vstack(outer_A, axis=0)

  return A