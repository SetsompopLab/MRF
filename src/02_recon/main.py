import os
import sys
import time
import argparse
import numpy as np
import sigpy as sp
import sigpy.mri as mr
from scipy.interpolate import interpn

from sigpy.mri.dcf import pipe_menon_dcf

import mrf
import prox
import optalg

from load_data import load_data

os.environ['OMP_NUM_THREADS'] = '80'

def main(args):

  dev = sp.Device(args.dev)
  xp = dev.xp

  mvd = lambda x: sp.to_device(x, dev)
  mvc = lambda x: sp.to_device(x, sp.cpu_device)

  print("> Loading data... ", end="", flush=True)
  start_time = time.perf_counter()
  (trj, ksp, phi) = load_data(args.trj, args.ksp, args.phi, args.rnk, \
                              args.akp, args.ptt)
  end_time = time.perf_counter()
  print("done. Time taken: %0.2f seconds." % (end_time - start_time), \
        flush=True)
  if trj.shape[0] == 2:
    (sx, sy) = [args.mtx]*2
    sz = 1
  elif trj.shape[0] == 3:
    (sx, sy, sz) = [args.mtx]*3

  print("> Dimensions: ", flush=True)
  print(">> trj:", trj.shape, flush=True)
  print(">> ksp:", ksp.shape, flush=True)
  print(">> phi:", phi.shape, flush=True)

  if args.ccm is not None:
    print("> Coil processing... ",
          end="", flush=True)
    start_time = time.perf_counter()
    with dev:
      ccm      = xp.load( args.ccm)
      shape    = list(ksp.shape)
      shape[0] = ccm.shape[0]

      ksp     = np.reshape(ksp, (ksp.shape[0], -1))
      new_ksp = np.zeros((shape[0], ksp.shape[-1]), dtype=ksp.dtype)
      batch   = 1024
      for k in range(0, ksp.shape[-1], batch):
        new_ksp[:, k:(k + batch)] = mvc(ccm @ mvd(ksp[:, k:(k + batch)]))
      del ksp
    ksp = np.reshape(new_ksp, shape)
    end_time = time.perf_counter()
    print("done. Time taken: %0.2f seconds." % (end_time - start_time), \
          flush=True)

  elif args.ccn is not None:
    kmat = np.reshape(ksp[:, :64, ...], (ksp.shape[0], -1))
    (u, _, _) = np.linalg.svd(kmat, full_matrices=False)
    ccm = mvd(u[:, :args.ccn].conj().T)
    shape = list(ksp.shape)
    shape[0] = args.ccn
    with dev:
      ksp     = np.reshape(ksp, (ksp.shape[0], -1))
      new_ksp = np.zeros((shape[0], ksp.shape[-1]), dtype=ksp.dtype)
      batch   = 1024
      for k in range(0, ksp.shape[-1], batch):
        new_ksp[:, k:(k + batch)] = mvc(ccm @ mvd(ksp[:, k:(k + batch)]))
      del ksp
    ksp = np.reshape(new_ksp, shape)

    
  if args.akp == False and args.shf is not None:
    shifts = np.load(args.shf)
    print("> Loading shifts (in mm): "
          "(x: %0.3f, y: %0.3f, z: %0.3f)" % tuple(shifts),
          flush=True)
    print("> Shifting ksp... ", end="", flush=True)
    start_time = time.perf_counter()
    px = np.exp(-1j * 2 * np.pi * trj[0, ...] * shifts[0])[None, ...]
    py = np.exp(-1j * 2 * np.pi * trj[1, ...] * shifts[1])[None, ...]
    if trj.shape[0] == 2:
      pz = 1
    if trj.shape[0] == 3:
      pz = np.exp(-1j * 2 * np.pi * trj[2, ...] * shifts[2])[None, ...]
    ksp = ksp * px * py * pz
    end_time = time.perf_counter()
    print("done. Time taken: %0.2f seconds." % (end_time - start_time),
          flush=True)

  if args.svk is not None:
    print("> Saving processed k-space... ", end="", flush=True)
    start_time = time.perf_counter()
    np.save( args.svk, ksp)
    end_time = time.perf_counter()
    print("done. Time taken: %0.2f seconds." % (end_time - start_time),
          flush=True)

  # Scaling trajectory
  trj[0, ...] = trj[0, ...] * sx
  trj[1, ...] = trj[1, ...] * sy
  if trj.shape[0] == 3:
    trj[2, ...] = trj[2, ...] * sz

  # Coil calibration.
  if args.mps is not None and os.path.isfile( args.mps):
    print("> Loading maps... ", end="", flush=True)
    start_time = time.perf_counter()
    with dev:
      mps = xp.load( args.mps)
    end_time = time.perf_counter()
    print("done. Time taken: %0.2f seconds." % (end_time - start_time),
          flush=True)
  else:
    print("> Coil Estimation... ", end="", flush=True)
    start_time = time.perf_counter()
    calib_ksp = ksp[   :, ...].transpose((1, 2, 3, 0)).T
    calib_trj = trj[::-1, ...].T
    nc = calib_ksp.shape[0]
    if args.mal=='espirit':
      print("Using ESPIRiT", flush=True)
      with dev:
        im = sp.nufft_adjoint(calib_ksp,calib_trj,oshape=(nc,sz,sy,sx),oversamp=2)
        calib_ksp = sp.to_device(sp.resize(sp.fft(im,axes=(1,2,3)),(nc,sz,sy,sx)),dev)
        mps = mr.app.EspiritCalib(calib_ksp, \
                        device=dev, \
                        show_pbar=False,crop=0.7,thresh=0.03).run()
        calib_ksp = sp.to_device(calib_ksp, dev)
    elif args.mal=='jsense':
      print("Using JSENSE", flush=True)
      mps = mr.app.JsenseRecon(calib_ksp, coord=calib_trj, \
                               device=dev, img_shape=(sz, sy, sx),
                               show_pbar=False).run()
    end_time = time.perf_counter()
    print("done. Time taken: %0.2f seconds." % (end_time - start_time),
          flush=True)

    if args.mps is not None:
      print("> Saving maps... ", end="", flush=True)
      start_time = time.perf_counter()
      xp.save( args.mps, mps)
      end_time = time.perf_counter()
      print("Time taken: %0.2f seconds." % (end_time - start_time), flush=True)

  if args.nco is not None:
    mps = mps[:args.nco,...]
    ksp = ksp[:args.nco,...]

  # Preparing data for full reconstruction.
  print("> Permuting data... ", flush=True, end="")
  start_time = time.perf_counter()
  trj = trj[::-1, ...].T
  ksp = np.transpose(ksp, (1, 2, 3, 0)).T
  phi = phi @ sp.fft(np.eye(phi.shape[-1]), axes=(0,))
  phi = phi.T
  end_time = time.perf_counter()
  print("done. Time taken: %0.2f seconds." % (end_time - start_time),
        flush=True)


  if args.dcf is not None and os.path.isfile( args.dcf):
    # Density compensation.
    print("> Loading DCF... ", end="", flush=True)
    start_time = time.perf_counter()
    with dev:
      dcf = xp.load( args.dcf).astype(xp.float32)
    end_time = time.perf_counter()
    print("done. Time taken: %0.2f seconds." % (end_time - start_time),
          flush=True)
  elif args.dcf is not None and not os.path.isfile( args.dcf):
    print("> Estimating and saving Pipe-Menon DCF.", flush=True)
    start_time = time.perf_counter()
    with dev:
      dcf = pipe_menon_dcf(trj, img_shape=mps.shape[1:], device=dev,
                           show_pbar=False).real.astype(xp.float32)
      xp.save( args.dcf, dcf)

  sqrt_dcf = None
  if args.dcf is not None:
    with dev:
      dcf /= xp.linalg.norm(dcf.ravel(), ord=xp.inf)
      sqrt_dcf = xp.sqrt(dcf)
      del dcf


  if args.bmp is not None:
    with dev:
      dct_b0 = {\
        'b0_arr' : xp.load( args.bmp),\
        'dt' : args.rdt,\
        'num_segments' : args.seg  }
  else:
    dct_b0 = None


  print("> Pre-reconstruction dimensions: ", flush=True)
  print(">> trj:", trj.shape, flush=True)
  print(">> ksp:", ksp.shape, flush=True)
  print(">> phi:", phi.shape, flush=True)
  print(">> mps:", mps.shape, flush=True)
  if args.dcf is not None:
    print(">> sqrt_dcf:", sqrt_dcf.shape, flush=True)
  if args.bmp is not None:
    print(">> b0_arr:", dct_b0['b0_arr'].shape, flush=True)

  with dev:
    phi = sp.to_device(phi, dev)
    mps = sp.to_device(mps, dev)
    trj = sp.to_device(trj, dev)
    if args.dcf is not None:
      sqrt_dcf = sp.to_device(sqrt_dcf, dev)
    ksp = ksp/np.linalg.norm(ksp)
    A = mrf.linop(trj, phi, mps, sqrt_dcf,dct_b0)
    if args.dcf is not None:
      ksp = sp.to_device(sqrt_dcf[None, ...], sp.cpu_device) * ksp
      ksp = ksp/np.linalg.norm(ksp)

    # Full reconstruction.
    init = None
    if args.int is not None:
      print("> Loading initial reconstruction... ", flush=True, end="")
      start_time = time.perf_counter()
      init = np.load( args.int).T.astype(np.complex64)
      end_time = time.perf_counter()
      print("done. Time taken: %0.2f seconds." % (end_time - start_time),
            flush=True)

    if args.a:
      print("> Adjoint Reconstruction... " , end="",
            flush=True)
      start_time = time.perf_counter()
      recon = A.H*ksp
      end_time = time.perf_counter()
      print("done. Time taken: %0.2f seconds." % (end_time - start_time), \
            flush=True)
    if args.c:
      print("> LSQ Reconstruction (lambda: %0.3e)... " % (args.lam), end="",
            flush=True)
      start_time = time.perf_counter()
      recon = sp.app.LinearLeastSquares(A, ksp, x=init, max_iter=args.mit,
                                        lamda=args.lam, show_pbar=False).run()
      end_time = time.perf_counter()
      print("done. Time taken: %0.2f seconds." % (end_time - start_time), \
            flush=True)

    if args.p:
      if args.pdg <= 0:
        print("> FISTA Reconstruction:", flush=True)
      else:
        print("> Poly. Precond. FISTA Reconstruction:", flush=True)
      start_time = time.perf_counter()
      LL = args.eig if args.eig is not None else \
           sp.app.MaxEig(A.N, dtype=xp.complex64, device=sp.cpu_device,
                         show_pbar=False).run() * 1.01
      print(">> Maximum eigenvalue estimated:", LL, flush=True)
      A = np.sqrt(1/LL) * A
      if args.blk > 0:
        stride = args.blk if args.str is None else args.str
        print(f">> LLR parameters:", flush=True)
        print(f">>> Block size: {args.blk}",       flush=True)
        print(f">>> Stride:     {stride}",         flush=True)
        print(f">>> Lambda:     %0.2e" % args.lam, flush=True)
        proxg = prox.LLR(A.ishape, args.lam, args.blk, args.dev,
                         stride=stride)
      else:
        print(f">> Wavelet with lambda %e." % args.lam)
        proxg = prox.L1Wav(A.ishape, args.lam, axes=(-1, -2, -3))
      (recon, ptol) = optalg.unconstrained(args.mit, A, ksp, proxg,
                                           pdeg=args.pdg,
                                           init=init)
      end_time = time.perf_counter()
      print(">> Time taken: %0.2f seconds." % (end_time - start_time), \
            flush=True)

    recon = mvc(recon).T

  print("> Saving reconstruction... ", end="", flush=True)
  start_time = time.perf_counter()
  np.save( args.res, recon)
  end_time = time.perf_counter()
  print("done. Time taken: %0.2f seconds." % (end_time - start_time), \
         flush=True)

def create_arg_parser():
  parser = argparse.ArgumentParser(description="MRF Reconstruction.")

  # Required parameters.
  parser.add_argument("--trj", type=str, required=True,                  \
    help="Trajectory.")
  parser.add_argument("--ksp", type=str, required=True,                  \
    help="k-space data.")
  parser.add_argument("--phi", type=str, required=True,                  \
    help="Temporal basis")
  parser.add_argument("--res", type=str, required=True,                  \
    help="Location to save the result.")

  # Reconstruction options.
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('-c', action='store_true',                          \
    help="Conjugate Gradient Reconstruction.")
  group.add_argument('-p', action='store_true',                          \
    help="Polynomial Preconditioned Reconstruction.")
  group.add_argument('-a', action='store_true',                          \
    help="Adjoint reconstruction.")

  # Optional parameters.
  parser.add_argument("--lrr", type=int, required=False, default=None,   \
    help="If set, reduce readout size to this.")
  parser.add_argument("--akp", action='store_true',                      \
    help="If set, assumes k-space has been pre-processed.")
  parser.add_argument("--int", type=str, required=False, default=None,   \
    help="Initialize reconstruction.")
  parser.add_argument("--ccm", type=str, required=False, default=None,   \
    help="Coil processing matrix.")
  parser.add_argument("--ccn", type=int, required=False, default=None,    \
     help="Number of virtual coils (if not using ccm).")
  parser.add_argument("--mps", type=str, required=False, default=None,   \
    help="If passed and file exists, load sensitivity maps. If passed "
         "and file does not exist, save sensitivity maps.")
  parser.add_argument("--mal", type=str, required=False, default='espirit',   \
    help="Sensitivity maps algorithm. either 'jsense' or 'espirit'")
  parser.add_argument("--bmp", type=str, required=False, default=None,   \
    help="If passed and file exists, load B0-map (in Hz).")
  parser.add_argument("--seg", type=int, required=False, default=6,   \
    help="Number of segments to use for B0 correction.")
  parser.add_argument("--rdt", type=float, required=False, default=4e-6,   \
    help="Readout sampling dt (in seconds).")
  parser.add_argument("--rnk", type=int, required=False, default=5,      \
    help="Rank of temporal subspace.")
  parser.add_argument("--mtx", type=int, required=False, default=256,    \
    help="Matrix size: [mtx, mtx, mtx]")
  parser.add_argument("--ptt", type=int, required=False, default=10,     \
    help="Number of readout points to throw away.")
  parser.add_argument("--dev", type=int, required=False, default=0,      \
    help="Device to use for reconstruction.")
  parser.add_argument("--mit", type=int, required=False, default=40,     \
    help="Number of iterations for reconstruction.")
  parser.add_argument("--nco", type=int, required=False, default=None,   \
    help="Number of coils to use (after coil compression).")
  parser.add_argument("--svk", type=str, required=False, default=None,   \
    help="Save k-space used for reconstruction.")
  parser.add_argument("--lam", type=float, required=False, default=0,    \
    help="Regularization value for specified reconstruction.")
  parser.add_argument("--dcf", type=str, required=False, default=None,   \
    help="Use DCF. Load if file exists, save if not.")

  # FOV options.
  parser.add_argument("--shf", type=str, required=False, default=None,   \
    help="(x, y, z) shifts. If NOT set, perform AutoFOV.")

  # Iterative reconstruction options.
  parser.add_argument("--pdg", type=int, required=False, default=0,      \
    help="(If \"-p\" is set) Degree of polynomial preconditioner.")
  parser.add_argument("--blk", type=int, required=False, default=8,      \
    help="(If \"-p\" is set) LLR block size.")
  parser.add_argument("--str", type=int, required=False, default=None,   \
    help="(If \"-p\" is set) Stride of LLR.")
  parser.add_argument("--eig", type=float, required=False, default=None, \
    help="(If \"-p\" is set) Lipchitz constant, if known.")

  return parser

if __name__ == "__main__":
  start_time = time.perf_counter()
  args = create_arg_parser().parse_args(sys.argv[1:])
  main(args)
  end_time = time.perf_counter()
  print("> Total time: %0.2f seconds." % (end_time - start_time))
