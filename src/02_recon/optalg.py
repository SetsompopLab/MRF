import numpy as np
import sigpy as sp
import time
import optpoly

def unconstrained(num_iters, A, b, proxg, pdeg=None, init=None):

  device = sp.get_device(b)
  xp = device.xp

  P = sp.linop.Identity(A.ishape) if pdeg <= 0 else  \
      optpoly.create_polynomial_preconditioner(pdeg, A.N, 0, 1, norm="l_2")

  with device:

    AHb = A.H(b).astype(xp.complex64)

    if type(init) == type(None):
      print(">> Starting reconstruction from AHb.", flush=True)
      x = AHb.copy()
    else:
      print(">> Starting reconstruction from init.", flush=True)
      x = sp.to_device(init.copy(), sp.get_device(AHb))
    z = x.copy()

    if num_iters <= 0:
      return (x, -1)

    print(">> Starting iterations: ", flush=True)
    for k in range(0, num_iters):
      print(">>> Starting iteration %03d... " % k, end="", flush=True)
      start_time = time.perf_counter()
      x_old = x.copy()
      x     = z.copy()

      gr    = A.N(x) - AHb
      x     = proxg(1, x - P(gr))
      if k == 0:
        z = x
      else:
        step  = k/(k + 3)
        z     = x + step * (x - x_old)
      end_time = time.perf_counter()
      print("done. Time taken: %0.2f seconds." % (end_time - start_time),
            flush=True)
    ptol = 100 * xp.linalg.norm(x_old - x)/xp.linalg.norm(x)
    print(">> Iterative tolerance percentage achieved: %0.2f" % ptol,
          flush=True)
    return (x, ptol)
