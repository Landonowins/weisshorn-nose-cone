import sys, time, numpy as np
import euler_axi as e
shape, f, M, rn = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
X,Y,ish = e.make_grid(shape, f, rn=rn)
s = e.EulerAxi(e.Mesh(X,Y,ish), M)
t=time.time(); cd = s.run(max_iter=12000, check=500); t=time.time()-t
print(f'{shape} f={f} M={M} rn={rn}: cd={cd:.5f} its={s.history[-1][0]} t={t:.0f}s hist={[round(h[1],5) for h in s.history[-5:]]}', flush=True)
