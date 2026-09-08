import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

# --- Grid Parameters ---
Nx, Ny = 50, 50#200, 200  
Nt = 10#100
T, L = 1.0, 2.5
dx, dy, dt = (2*L)/Nx, (2*L)/Ny, T/Nt
num_s = Nx * Ny

# --- Physics ---
nu, kappa, alpha = 0.5, 1.0, 2
x = np.linspace(-L, L, Nx, endpoint=False)
y = np.linspace(-L, L, Ny, endpoint=False)
xx, yy = np.meshgrid(x, y)
V_flat = (2.0 * (xx**2 - 1.0)**2 + 0.3 * xx + 0.5 * yy**2).flatten()
m0_flat = np.exp(-(xx**2 + yy**2) / (2 * 0.6**2)).flatten()
m0_flat /= (np.sum(m0_flat) * dx * dy)

# --- Operators ---
I_s = sp.eye(num_s, format='csc')
Z_s = sp.csc_matrix((num_s, num_s)) # <-- The missing zero matrix block

Dxf_1d = sp.diags([-1, 1, 1], [0, 1, -(Nx-1)], shape=(Nx, Nx)) / dx
Dxb_1d = sp.diags([-1, 1, -1], [-1, 0, Nx-1], shape=(Nx, Nx)) / dx
Dxf = sp.kron(sp.eye(Ny), Dxf_1d, format='csc')
Dxb = sp.kron(sp.eye(Ny), Dxb_1d, format='csc')
Dyf_1d = sp.diags([-1, 1, 1], [0, 1, -(Ny-1)], shape=(Ny, Ny)) / dy
Dyb_1d = sp.diags([-1, 1, -1], [-1, 0, Ny-1], shape=(Ny, Ny)) / dy
Dyf = sp.kron(Dyf_1d, sp.eye(Nx), format='csc')
Dyb = sp.kron(Dyb_1d, sp.eye(Nx), format='csc')
Lx_1d = sp.diags([1, -2, 1, 1, 1], [-1, 0, 1, Nx-1, -(Nx-1)], shape=(Nx, Nx)) / (dx**2)
Ly_1d = sp.diags([1, -2, 1, 1, 1], [-1, 0, 1, Ny-1, -(Ny-1)], shape=(Ny, Ny)) / (dy**2)
L_2d = (sp.kron(sp.eye(Ny), Lx_1d) + sp.kron(Ly_1d, sp.eye(Nx))).tocsc()

def build_global_system(W, nu_val):
    U = W[:(Nt+1)*num_s].reshape((Nt+1, num_s))
    M = W[(Nt+1)*num_s:].reshape((Nt+1, num_s))
    F_U, F_M = np.zeros_like(U), np.zeros_like(M)
    A_UU, A_UM, A_MU, A_MM = [[None]*(Nt+1) for _ in range(Nt+1)], [[None]*(Nt+1) for _ in range(Nt+1)], [[None]*(Nt+1) for _ in range(Nt+1)], [[None]*(Nt+1) for _ in range(Nt+1)]
    
    # --- Corrected Boundary Conditions padded with Z_s ---
    F_U[Nt], A_UU[Nt][Nt], A_UM[Nt][Nt], A_MU[Nt][Nt] = U[Nt] - V_flat, I_s, Z_s, Z_s
    F_M[0], A_MM[0][0], A_UM[0][0], A_MU[0][0] = M[0] - m0_flat, I_s, Z_s, Z_s
    
    for n in range(Nt):
        un, un_next, mn_next = U[n], U[n+1], M[n+1]
        dpx, dmx, dpy, dmy = Dxf.dot(un), Dxb.dot(un), Dyf.dot(un), Dyb.dot(un)
        F_U[n] = (un - un_next)/dt - nu_val*L_2d.dot(un) + 0.5*(np.minimum(dpx,0)**2 + np.maximum(dmx,0)**2 + np.minimum(dpy,0)**2 + np.maximum(dmy,0)**2) - (V_flat + kappa * (np.maximum(mn_next, 0)**alpha))
        A_UU[n][n] = (1.0/dt)*I_s - nu_val*L_2d + sp.diags(np.minimum(dpx,0))@Dxf + sp.diags(np.maximum(dmx,0))@Dxb + sp.diags(np.minimum(dpy,0))@Dyf + sp.diags(np.maximum(dmy,0))@Dyb
        A_UU[n][n+1], A_UM[n][n+1] = -(1.0/dt)*I_s, sp.diags(-kappa * alpha * (np.maximum(mn_next, 0)**(alpha-1)))
    
    for n in range(1, Nt+1):
        mn, mn_prev, un_prev = M[n], M[n-1], U[n-1]
        dpx, dmx, dpy, dmy = Dxf.dot(un_prev), Dxb.dot(un_prev), Dyf.dot(un_prev), Dyb.dot(un_prev)
        pminx, pmaxx, pminy, pmaxy = np.minimum(dpx,0), np.maximum(dmx,0), np.minimum(dpy,0), np.maximum(dmy,0)
        F_M[n] = (mn - mn_prev)/dt - nu_val*L_2d.dot(mn) - (Dxb.dot(mn*pminx) + Dxf.dot(mn*pmaxx) + Dyb.dot(mn*pminy) + Dyf.dot(mn*pmaxy))
        A_MM[n][n] = (1.0/dt)*I_s - nu_val*L_2d - (Dxb@sp.diags(pminx) + Dxf@sp.diags(pmaxx) + Dyb@sp.diags(pminy) + Dyf@sp.diags(pmaxy))
        A_MM[n][n-1], A_MU[n][n-1] = -(1.0/dt)*I_s, -(Dxb@sp.diags(mn*(dpx<0))@Dxf + Dxf@sp.diags(mn*(dmx>0))@Dxb + Dyb@sp.diags(mn*(dpy<0))@Dyf + Dyf@sp.diags(mn*(dmy>0))@Dyb)
    
    J = sp.bmat([[sp.bmat(A_UU), sp.bmat(A_UM)], [sp.bmat(A_MU), sp.bmat(A_MM)]], format='csc')
    return J, np.concatenate([F_U.flatten(), F_M.flatten()])

W = np.concatenate([np.tile(V_flat, (Nt+1, 1)).flatten(), np.tile(m0_flat, (Nt+1, 1)).flatten()])
print(f"--- GLOBAL NEWTON DIRECT STUDY (Nx={Nx}, Nt={Nt}) ---")
t_build = time.time()
J, F = build_global_system(W, nu)
print(f"Matrix Assembly Time: {time.time() - t_build:.2f}s")
print(f"Total DOFs:      {J.shape[0]}")
print(f"Jacobian NNZ:    {J.nnz}")

print("Attempting LU Factorization (Expect MemoryError/Killed)...")
t0 = time.time()
try:
    lu = spla.splu(J)
    solve_time = time.time() - t0
    print(f"Factor NNZ:      {lu.nnz}")
    print(f"Fill-in ratio:   {lu.nnz/J.nnz:.2f}x")
    print(f"Solve Time (s):  {solve_time:.6f}")
    print(f"Est. Factor RAM: {lu.nnz * 8 / 1e9:.4f} GB")
except Exception as e:
    print(f"\n>>> DIRECT SOLVE FAILED: {e}")

