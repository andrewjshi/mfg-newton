import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time
import sys

# ==========================================
# UTILITY: DUAL LOGGING (CONSOLE + FILE)
# ==========================================
class Logger(object):
    def __init__(self, filename="experiment1_results.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger()

# ==========================================
# SECTION 1: PARAMETERS AND GRID
# ==========================================
sigma = 1.0              
nu = 0.5 * sigma**2      
mu = 0.0                 
d = 1 

lam_screenshot = -d + (d/2.0) * np.log(2.0 / (np.pi * sigma**4))

width = 2.0              
xmin, xmax = mu - width, mu + width 
Nx = 200                

# Iteration Parameters - RESTORED TO STRICT PRECISION
PICARD_MAX_ITER = 15000     
PICARD_TOL = 1e-8        
NEWTON_MAX_ITER = 20     
NEWTON_TOL = 1e-10       
MASS_FLOOR = 1e-15       

# Gentle slope to overcome the 1e-4 stiffness barrier
all_rhos = [1e-1, 1e-2, 5e-3, 1e-3, 5e-4, 2e-4, 1e-4]

L = xmax - xmin          
Dx = L / Nx              
x = np.linspace(xmin, xmax, Nx + 1) 
num_x = x.size

# ==========================================
# SECTION 2: DISCRETE SPATIAL OPERATORS
# ==========================================
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

# ==========================================
# SECTION 3: ROBUST DISCOUNTED SOLVER
# ==========================================
def solve_stationary_discounted(rho, u_guess, m_guess, damping):
    u_curr = u_guess.copy()
    m_curr = m_guess.copy()
    dt_pseudo = 10.0 
    
    for it in range(PICARD_MAX_ITER):
        dp, dm = D_plus @ u_curr, D_minus @ u_curr
        p_min, p_max = np.minimum(dp, 0), np.maximum(dm, 0)
        J_G = (sp.diags(p_min) @ D_plus + sp.diags(p_max) @ D_minus).tocsr()
        Adv_Op = J_G.transpose()
        
        A_FP = (I + dt_pseudo * (-nu * Laplacian + Adv_Op)).tolil()
        A_FP[0, :] = 0; A_FP[0, 0] = 1.0; A_FP[0, 1] = -1.0
        A_FP[-1, :] = 0; A_FP[-1, -1] = 1.0; A_FP[-1, -2] = -1.0
        
        rhs_m = m_curr.copy()
        rhs_m[0] = 0.0; rhs_m[-1] = 0.0
        
        m_next = spla.spsolve(A_FP.tocsr(), rhs_m)
        m_next = np.maximum(m_next, MASS_FLOOR)
        m_next /= (np.sum(m_next) * Dx)  
        
        u_next = u_curr.copy()
        f_val = -np.log(m_next)
        
        for _ in range(NEWTON_MAX_ITER):
            dp, dm = D_plus @ u_next, D_minus @ u_next
            H_val = 0.5 * np.minimum(dp, 0)**2 + 0.5 * np.maximum(dm, 0)**2
            
            F = rho * u_next - nu * Laplacian @ u_next + H_val - f_val
            F[0] = u_next[0] - u_next[1]
            F[-1] = u_next[-1] - u_next[-2]
            
            if np.linalg.norm(F, np.inf) < NEWTON_TOL: break
                
            p_min, p_max = np.minimum(dp, 0), np.maximum(dm, 0)
            dH_dU = sp.diags(p_min) @ D_plus + sp.diags(p_max) @ D_minus
            
            J = rho * I - nu * Laplacian + dH_dU
            J = J.tolil()
            J[0, :] = 0; J[0, 0] = 1.0; J[0, 1] = -1.0
            J[-1, :] = 0; J[-1, -1] = 1.0; J[-1, -2] = -1.0
            
            u_next -= spla.spsolve(J.tocsr(), F)
            
        err_u = np.linalg.norm(u_next - u_curr, np.inf)
        err_m = np.linalg.norm(m_next - m_curr, np.inf)
        
        u_curr = (1 - damping) * u_curr + damping * u_next
        m_curr = (1 - damping) * m_curr + damping * m_next
        
        if err_u < PICARD_TOL and err_m < PICARD_TOL: break
            
    return u_curr, m_curr, it

# ==========================================
# SECTION 4: CONDUCT EXPERIMENT 1 
# ==========================================
print("="*75)
print(f" Reference Constant from Screenshot: {lam_screenshot:.8f}")
print("="*75)
print("Running Gentle Asymptotic Continuation...")

u_guess = np.zeros(num_x)
m_guess = np.ones(num_x) / L 
results = {}

for i in range(len(all_rhos)):
    rho = all_rhos[i]
    start_time = time.time()
    
    # Adaptive damping: Heavy damping (slower steps) for highly stiff rho
    current_damping = 0.5 if rho >= 1e-3 else 0.1
    
    u_rho, m_rho, iters = solve_stationary_discounted(rho, u_guess, m_guess, damping=current_damping)
    results[rho] = (u_rho, m_rho)
    lam_approx = np.mean(rho * u_rho)
    
    print(f"Solved rho = {rho:<7.1e} | Damping: {current_damping:.1f} | Iters: {iters:<4} | Time: {time.time()-start_time:.2f}s | lambda_est: {lam_approx:.8f}")
    
    if i < len(all_rhos) - 1:
        rho_next = all_rhos[i+1]
        shift = lam_approx * (1.0 / rho_next - 1.0 / rho)
        u_guess = u_rho + shift
        m_guess = m_rho.copy()

u_1e4, m_1e4 = results[1e-4]
dp, dm = D_plus @ u_1e4, D_minus @ u_1e4
H_val_1e4 = 0.5 * np.minimum(dp, 0)**2 + 0.5 * np.maximum(dm, 0)**2
lambda_h = np.sum(m_1e4 * (-np.log(m_1e4) + H_val_1e4)) * Dx

print("\n" + "="*75)
print(f" GROUND TRUTH VERIFICATION")
print("="*75)
print(f"True Discrete Constant (lambda_h) via Energy Integral: {lambda_h:.8f}")
print(f"Difference from Screenshot Reference:                {abs(lambda_h - lam_screenshot):.4e}")

# ==========================================
# SECTION 5: PLOTTING AND SUMMARY
# ==========================================
fig = plt.figure(figsize=(10, 6))
table_data = []

# Only plot the main targets to keep the graph clean
plot_rhos = [1e-1, 1e-2, 1e-3, 1e-4]

for rho in all_rhos:
    u_rho, _ = results[rho]
    rho_u = rho * u_rho 
    max_err = np.max(np.abs(rho_u - lambda_h))
    
    if rho in plot_rhos:
        table_data.append((rho, max_err))
        plt.plot(x, rho_u, linewidth=2, label=f'$\\rho = {rho:.0e}$')

plt.axhline(lambda_h, color='k', linestyle='--', linewidth=2, label=f'$\\lambda_h$ (Discrete Ergodic)')
plt.title('Convergence of Scaled Value Function $\\rho u_\\rho$ to $\\lambda_h$', fontsize=14)
plt.xlabel('Space (x)', fontsize=12)
plt.ylabel('$\\rho u_\\rho(x)$', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("experiment1_plot.png")

print("\n" + "="*50)
print(f" EXPERIMENT 1 RESULTS (FIXED GRID Nx={Nx}, L={L})")
print("="*50)
print(f"{'Discount (rho)':<15} | {'max |rho*u - lambda_h|':<25}")
print("-" * 50)
for r, err in table_data:
    print(f"{r:<15.1e} | {err:<25.4e}")
print("="*50)
print(f"\nResults saved to experiment1_results.txt and experiment1_plot.png")
