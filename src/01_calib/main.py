import sys
import time
import argparse

import numpy as np
import sigpy as sp

from rovir import rovir
from autofov import autofov


def get_gre_ksp(raw):
  # ksp after reshape: (kx, nc, ky, z)
  ksp = np.reshape(raw, (raw.shape[0], raw.shape[1], 64, 64))
  ksp = np.transpose(ksp, (0, 2, 3, 1))
  tmp = np.zeros_like(ksp)
  tmp[:, :, 0::2, :] = ksp[:, :, :32, :]
  tmp[:, :, 1::2, :] = ksp[:, :, 32:, :]
  img = sp.ifft(tmp, axes=(0, 1))
  img = np.transpose(img, (2, 1, 0, 3))
  img = img[:, ::-1, ::-1, :]
  return sp.fft(img, axes=(0, 1, 2))


def main(args):
  if args.ksp is not None:
    ## Load data.
    raw = np.load( args.ksp)
    nc = np.shape(raw)[1]

    if raw.shape[0] != 64 or raw.shape[2] != 4096:
      raise Exception("Currently, only GRE calibration supported.")
    print("> Assumed GRE parameters:", flush=True)
    print("\tMatrix size:  64 x  64 x  64", flush=True)
    print("\tFOV:         440 x 440 x 440 mm^3", flush=True)

    ksp = get_gre_ksp(raw)
    img = sp.ifft(ksp, axes=(0, 1, 2))

    mtx_shape = list(img.shape[:3])

    # AutoFOV
    if args.shf is not None:
      shifts = autofov(np.linalg.norm(img, axis=3)**2, 440/64, FOV_target=args.tfv)
      np.save( args.shf, shifts + (np.array(args.ofs) * (440/64)))

      print(">> Applying shifts... ", end="", flush=True)
      kx = np.arange(-img.shape[0]/2, img.shape[0]/2, 1)/img.shape[0]
      ky = np.arange(-img.shape[0]/2, img.shape[0]/2, 1)/img.shape[0]
      kz = np.arange(-img.shape[0]/2, img.shape[0]/2, 1)/img.shape[0]
      px = np.exp(-1j * 2 * np.pi * kx * (float(shifts[0])/(440/64)))
      py = np.exp(-1j * 2 * np.pi * ky * (float(shifts[1])/(440/64)))
      pz = np.exp(-1j * 2 * np.pi * kz * (float(shifts[2])/(440/64)))
      ksp = ksp * px[:, None, None, None] * \
                  py[None, :, None, None] * \
                  pz[None, None, :, None]
      img = sp.ifft(ksp, axes=(0, 1, 2))

    # Load noise matrix.
    nmat = np.load( args.nse)

    if args.idx is not None:
      idx = np.load(args.idx)
      nmat = nmat[idx,...]
      nc = len(idx)
      ksp = ksp[..., idx]
      img = img[..., idx]

    ############################### ROVIR ###############################
    # Link: https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.28706
    # Based on Section 2.3.
    #
    # 1. Pre-whiten k-space.
    # 2. Calculate A & B
    # 3. Calculate generalized eigenvalue decomposition.
    # 4. Select number of virtual coils.
    # 5. Normalize generalized eigenvectors to have unit l2-norm.
    # 6. Ortho-normalize chosen coils => virtual channels are whitened.

    # 1. Whitening.
    covm = (nmat @ nmat.conj().T)/(nmat.shape[1] - 1)
    wht  = np.linalg.pinv(np.linalg.cholesky(covm)) # Whitening matrix.
    img  = np.reshape((wht @ np.reshape(img, (-1, nc)).T).T, mtx_shape + [nc])

    # 2. Calculate A and B of ROVir.
    # FIXME: Currently one call is round and one is int (floor).
    roi = {}
    roi["x"] = np.arange(round((440 - args.tfv)/2 * (64/440)),
                          int((440 + args.tfv)/2 * (64/440)))
    roi["y"] = np.arange(round((440 - args.tfv)/2 * (64/440)),
                          int((440 + args.tfv)/2 * (64/440)))
    roi["z"] = np.arange(round((440 - args.tfv)/2 * (64/440)),
                          int((440 + args.tfv)/2 * (64/440)))
    rtf = None

    # 3. Calculate generalized eigenvalue decomposition.
    # 4. Select number of virtual coils.
    eig = rovir(img, roi, rtf)[:, :args.nrc]

    # 5. Normalize to unit l2-norm.
    for k in range(args.nrc):
      eig[:, k] = eig[:, k]/np.linalg.norm(eig[:, k])

    # 6. Ortho-normalize.
    (rcc, _, _) = np.linalg.svd(eig, full_matrices=False)

    # Coil processing matrix so far.
    ccm = rcc.T @ wht

    # Apply processing to k-space, and take SVD decomposition.
    ksp_shape = list(ksp.shape)
    ksp_shape[-1] = args.nsv

    ksp = np.reshape(ksp, (-1, nc))
    ksp = ccm @ ksp.T

    (u, _, _) = np.linalg.svd(ksp, full_matrices=False)
    u  = u[:, :args.nsv]
    uH = u.conj().T

    ksp = np.reshape((uH @ ksp).T, ksp_shape)
    ccm = uH @ ccm

    img = sp.ifft(ksp, axes=(0, 1, 2))

    ## Save results
    if args.rci is not None:
      np.save( args.rci, img)
    np.save( args.ccm, ccm)
  else:
    ccm=None
    shifts=[0,0,0]



def create_arg_parser():
  parser = argparse.ArgumentParser(description="Calibration.")

  # GRE parameters.
  parser.add_argument("--ksp", type=str, required=False,
    default=None, help="GRE k-space data.")
  parser.add_argument("--nse", type=str, required=False,
    default=None, help="GRE noise measurements.")
  parser.add_argument("--ccm", type=str, required=False,
    default=None, help="Location to save coil processing matrix.")


  # Optional arguments.
  parser.add_argument("--rci", type=str, required=False,
    default=None, help="Location to save ROVir coil images.")
  parser.add_argument("--nrc", type=int, required=False,
    default=40, help="Number of ROVir coils.")
  parser.add_argument("--nsv", type=int, required=False,
    default=4, help="Number of SVD coils.")
  parser.add_argument("--idx", type=str, required=False,
    default=None, help="Indecies of coils to use.")
  parser.add_argument("--shf", type=str, required=False,
    default=None, help="Location to save autoFOV shifts.")
  parser.add_argument("--tfv", type=int, required=False,
    default=256, help="Target FOV (mm)")
  parser.add_argument("--ofs", type=list, required=False,
    nargs="+", default=[0.5, -1, -1], help="Offset between calibration "
                                               "and MRF (in calibration "
                                               "pixels)")
  
  return parser


if __name__ == "__main__":
  start_time = time.perf_counter()
  args = create_arg_parser().parse_args(sys.argv[1:])
  main(args)
  end_time = time.perf_counter()
  print("> Total time: %0.2f seconds." % (end_time - start_time))
