import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time

# ==========================================
# SECTION 1: PARAMETERS AND EXPERIMENT SETTINGS
# ==========================================

# --- Physics Parameters ---
sigma = 1.0              
nu = 0.5 * sigma**2      
mu = 0.0                 

# --- Analytical Ergodic Constant ---
# lambda = -1.0 + 0.5 * log(2 / (pi * sigma^4))
lam_exact = -1.0 + 0.5 * np.log(2.0 / (np.pi * sigma**4))

# --- Picard and Newton Iteration Parameters ---
PICARD_MAX_ITER = 150    
PICARD_TOL = 1e-7        
DAMPING = 0.5            
NEWTON_MAX_ITER = 20     
NEWTON_TOL = 1e-10       
MASS_FLOOR = 1e-15       

# --- Experiment Settings ---
# We use a tiny rho to extract the exact discrete ergodic state
GROUND_TRUTH_RHO = 1e-8

# The optimally balanced sequence of (Domain L, Grid Nx)
# Nx grows by a factor of 4 while L grows by 1.
grid_configurations = [
    (4.0, 200),
    (5.0, 800),
    (6.0, 3200),
    (7.0, 12800)
]

# --- Saved File Names ---
plot_convergence_name = "experiment2_domain_grid_convergence.png"


# ==========================================
# SECTION 2: STATIONARY SOLVER FOR VARIABLE GRIDS
# ==========================================

def get_discrete_ergodic_constant(L, Nx):
    """Builds operators for a specific (L, Nx) and solves for lambda_h"""
    
    # 1. Grid Setup
    xmin, xmax = mu - L/2, mu + L/2
    Dx = L / Nx
    x = np.linspace(xmin, xmax, Nx + 1)
    num_x = x.size
    
    # 2. Discrete Spatial Operators
    I = sp.eye(num_x)
    
    data_p = np.array([-np.ones(num_x), np.ones(num_x)])
    D_plus = sp.spdiags(data_p, [0, 1], num_x, num_x).tolil()
    D_plus[-1, :] = 0 
    D_plus = D_plus.tocsr() / Dx

    data_m = np.array([-np.ones(num_x), np.ones(num_x)])
    D_minus = sp.spdiags(data_m, [-1, 0], num_x, num_x).tolil()
    D_minus[0, :] = 0 
    D_minus = D_minus.tocsr() / Dx

    Laplacian = D_plus @ D_minus
    
    # 3. Solver Initialization
    u_curr = np.zeros(num_x)
    m_curr = np.ones(num_x) / L 
    
    # 4. Iterative Solver
    for it in range(PICARD_MAX_ITER):
        # --- A. Solve Forward FP Equation given u_curr ---
        dp, dm = D_plus @ u_curr, D_minus @ u_curr
        p_min, p_max = np.minimum(dp, 0), np.maximum(dm, 0)
        J_G = (sp.diags(p_min) @ D_plus + sp.diags(p_max) @ D_minus).tocsr()
        Adv_Op = J_G.transpose()
        
        A_FP = (-nu * Laplacian + Adv_Op).tolil()
        
        # Neumann BCs for m (zero flux)
        A_FP[0, :] = 0; A_FP[0, 0] = -1.0; A_FP[0, 1] = 1.0
        A_FP[-1, :] = Dx # Mass Normalization
        
        rhs_m = np.zeros(num_x)
        rhs_m[-1] = 1.0  
        
        m_next = spla.spsolve(A_FP.tocsr(), rhs_m)
        m_next = np.maximum(m_next, MASS_FLOOR)
        
        # --- B. Solve Backward HJB Equation given m_next ---
        u_next = u_curr.copy()
        f_val = -np.log(m_next)
        
        for _ in range(NEWTON_MAX_ITER):
            dp, dm = D_plus @ u_next, D_minus @ u_next
            H_val = 0.5 * np.minimum(dp, 0)**2 + 0.5 * np.maximum(dm, 0)**2
            
            F = GROUND_TRUTH_RHO * u_next - nu * Laplacian @ u_next + H_val - f_val
            
            # Neumann BCs for u: u'(x) = 0
            F[0] = u_next[0] - u_next[1]
            F[-1] = u_next[-1] - u_next[-2]
            
            if np.linalg.norm(F, np.inf) < NEWTON_TOL: 
                break
                
            p_min, p_max = np.minimum(dp, 0), np.maximum(dm, 0)
            dH_dU = sp.diags(p_min) @ D_plus + sp.diags(p_max) @ D_minus
            
            J = GROUND_TRUTH_RHO * I - nu * Laplacian + dH_dU
            J = J.tolil()
            
            J[0, :] = 0; J[0, 0] = 1.0; J[0, 1] = -1.0
            J[-1, :] = 0; J[-1, -1] = 1.0; J[-1, -2] = -1.0
            
            u_next -= spla.spsolve(J.tocsr(), F)
            
        # --- C. Damping and Convergence Check ---
        err_u = np.linalg.norm(u_next - u_curr, np.inf)
        err_m = np.linalg.norm(m_next - m_curr, np.inf)
        
        u_curr = (1 - DAMPING) * u_curr + DAMPING * u_next
        m_curr = (1 - DAMPING) * m_curr + DAMPING * m_next
        
        if err_u < PICARD_TOL and err_m < PICARD_TOL:
            break
            
    # Because of Neumann BCs, rho * u is perfectly flat. 
    # Its mean is exactly lambda_h.
    lambda_h = np.mean(GROUND_TRUTH_RHO * u_curr)
    return lambda_h


# ==========================================
# SECTION 3: CONDUCT EXPERIMENT 2
# ==========================================

print("="*60)
print(f" Analytical Exact Ergodic Constant: {lam_exact:.8f}")
print("="*60)
print("Running simultaneous domain and grid refinement...")

results = []
errors = []
labels = []

for L_val, Nx_val in grid_configurations:
    start_time = time.time()
    
    # Compute the discrete ergodic constant for this specific grid
    lam_h = get_discrete_ergodic_constant(L_val, Nx_val)
    
    # Compare it to the true analytical constant
    err = abs(lam_h - lam_exact)
    
    compute_time = time.time() - start_time
    print(f"Solved L={L_val}, Nx={Nx_val:<5} | lam_h = {lam_h:.6f} | err = {err:.3e} | Time: {compute_time:.2f}s")
    
    results.append((L_val, Nx_val, lam_h, err))
    errors.append(err)
    labels.append(f"L={int(L_val)}\nNx={Nx_val}")


# ==========================================
# SECTION 4: PLOTTING AND OUTPUT
# ==========================================

# 1. Generate the Convergence Plot
plt.figure(figsize=(8, 6))
plt.semilogy(range(len(errors)), errors, 'bo-', linewidth=2, markersize=8)
plt.xticks(range(len(errors)), labels)
plt.title('Convergence to Analytical Ergodic Constant $\lambda$', fontsize=14)
plt.xlabel('Grid Configuration (Domain L, Resolution Nx)', fontsize=12)
plt.ylabel('Absolute Error $|\lambda_h - \lambda_{exact}|$', fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.tight_layout()
plt.savefig(plot_convergence_name)
plt.show()

# 2. Print LATEX-Ready Table
print("\n" + "="*50)
print(" EXPERIMENT 2 RESULTS (SIMULTANEOUS REFINEMENT)")
print("="*50)
print(f"{'Domain L':<10} | {'Grid Nx':<10} | {'|lambda_h - lambda_exact|':<25}")
print("-" * 50)
for L_val, Nx_val, _, err in results:
    print(f"{L_val:<10.1f} | {Nx_val:<10} | {err:<25.4e}")
print("="*50)
