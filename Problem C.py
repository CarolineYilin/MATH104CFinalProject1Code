import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Problem C: Nonlinear Gradient System ---
def f(t, y):
    return y * (1 - y**2)

# For the Taylor Method of order 2, we need f'(t,y)
# Using chain rule/product rule: f' = y' * (1 - y^2) + y * (-2 * y * y')
# Since y' = f(t, y), we substitute:
def df(t, y):
    y_prime = y * (1 - y**2)
    return y_prime * (1 - y**2) + y * (-2 * y * y_prime)

def exact_sol(t):
    # The analytical solution to y' = y(1 - y^2)
    # Using separation of variables and partial fractions
    y0 = 0.01
    return 1 / np.sqrt(1 + ((1/y0**2) - 1) * np.exp(-2 * t))

# --- Numerical Methods ---
#1. Euler
def euler(t, w, h):
    return w + h * f(t, w)
#2. taylor order 2
def taylor_order2(t, w, h):
    return w + h * f(t, w) + (h**2 / 2) * df(t, w)
#3.midpoint
def midpoint(t, w, h):
    return w + h * f(t + h/2, w + (h/2) * f(t, w))
#3.modified Euler's method
def modified_euler(t, w, h):
    return w + (h/2) * (f(t, w) + f(t + h, w + h * f(t, w)))
#4.Heun's method
def heun3(t, w, h):
    k1 = f(t, w)
    k2 = f(t + h/3, w + (h/3) * k1)
    k3 = f(t + 2*h/3, w + (2*h/3) * k2)
    return w + (h/4) * (k1 + 3 * k3)
#5.RK4
def rk4(t, w, h):
    k1 = h * f(t, w)
    k2 = h * f(t + h/2, w + k1/2)
    k3 = h * f(t + h/2, w + k2/2)
    k4 = h * f(t + h, w + k3)
    return w + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

# --- Multistep Method ---
def predictor_corrector_ab4_am3(t_vals, h, y0):
    n = len(t_vals)
    w = np.zeros(n)
    w[0] = y0
    
    # Generate starting values w1, w2, w3 using RK4
    for i in range(3):
        w[i+1] = rk4(t_vals[i], w[i], h)
        
    for i in range(3, n-1):
        t_i, t_im1, t_im2, t_im3 = t_vals[i], t_vals[i-1], t_vals[i-2], t_vals[i-3]
        f_i   = f(t_i, w[i])
        f_im1 = f(t_im1, w[i-1])
        f_im2 = f(t_im2, w[i-2])
        f_im3 = f(t_im3, w[i-3])
        
        # Predictor (Adams-Bashforth 4)
        w_pred = w[i] + (h/24) * (55*f_i - 59*f_im1 + 37*f_im2 - 9*f_im3)
        
        # Evaluator
        f_ip1 = f(t_vals[i+1], w_pred)
        
        # Corrector (Adams-Moulton 3)
        w[i+1] = w[i] + (h/24) * (9*f_ip1 + 19*f_i - 5*f_im1 + f_im2)
        
    return w

# --- Main Execution ---
a, b = 0, 2
y0 = 0.5
h = 0.2  # Test with step size 0.2 (can easily be changed to 0.1 or 0.05)

t_vals = np.arange(a, b + h, h)
exact_vals = exact_sol(t_vals)

results = {
    't': t_vals,
    'Exact': exact_vals,
    'Euler': np.zeros_like(t_vals),
    'Taylor2': np.zeros_like(t_vals),
    'Midpoint': np.zeros_like(t_vals),
    'Mod_Euler': np.zeros_like(t_vals),
    'Heun3': np.zeros_like(t_vals),
    'RK4': np.zeros_like(t_vals)
}

# Initialize w0
for key in results.keys():
    if key not in ['t', 'Exact']:
        results[key][0] = y0

# Compute One-Step Methods
for i in range(len(t_vals) - 1):
    t_i = t_vals[i]
    results['Euler'][i+1] = euler(t_i, results['Euler'][i], h)
    results['Taylor2'][i+1] = taylor_order2(t_i, results['Taylor2'][i], h)
    results['Midpoint'][i+1] = midpoint(t_i, results['Midpoint'][i], h)
    results['Mod_Euler'][i+1] = modified_euler(t_i, results['Mod_Euler'][i], h)
    results['Heun3'][i+1] = heun3(t_i, results['Heun3'][i], h)
    results['RK4'][i+1] = rk4(t_i, results['RK4'][i], h)

# Compute Predictor-Corrector
results['Pred_Corr'] = predictor_corrector_ab4_am3(t_vals, h, y0)

# Display Results DataFrame
df_results = pd.DataFrame(results)
print("Approximations (h = 0.2):")
print(df_results)

# Compute Absolute Errors
df_errors = pd.DataFrame({'t': t_vals})
for col in df_results.columns:
    if col not in ['t', 'Exact']:
        df_errors[col] = np.abs(df_results['Exact'] - df_results[col])

print("\nAbsolute Errors (h = 0.2):")
print(df_errors)


# plotting separately without Adams–Bashforth and Adams–Moulton
fig, axs = plt.subplots(2, 4, figsize=(15, 8)) # Creates a 2x4 grid
methods = ['Euler', 'Taylor2', 'Midpoint', 'Mod_Euler', 'Heun3', 'RK4', 'Pred_Corr']

# Loop through to plot each method on its own subplot
for i, method in enumerate(methods):
    row, col = i // 4, i % 4
    axs[row, col].plot(t_vals, exact_vals, 'k--', label='Exact')
    axs[row, col].plot(t_vals, results[method], label=method)
    axs[row, col].set_title(method)
    axs[row, col].legend()

plt.tight_layout()
plt.show()

# Plot Error Curves
plt.figure(figsize=(10, 6))
for col in df_errors.columns:
    if col != 't':
        plt.plot(df_errors['t'], df_errors[col], label=col)

plt.title('Absolute Errors for Problem A (h=0.2)')
plt.yscale('log') # Log scale is often best for visualizing drastically different error magnitudes
plt.xlabel('t')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(True)
plt.show()


#-----h=0.4-----

# --- Numerical Methods ---
#1. Euler
def euler(t, w, h):
    return w + h * f(t, w)
#2. taylor order 2
def taylor_order2(t, w, h):
    return w + h * f(t, w) + (h**2 / 2) * df(t, w)
#3.midpoint
def midpoint(t, w, h):
    return w + h * f(t + h/2, w + (h/2) * f(t, w))
#3.modified Euler's method
def modified_euler(t, w, h):
    return w + (h/2) * (f(t, w) + f(t + h, w + h * f(t, w)))
#4.Heun's method
def heun3(t, w, h):
    k1 = f(t, w)
    k2 = f(t + h/3, w + (h/3) * k1)
    k3 = f(t + 2*h/3, w + (2*h/3) * k2)
    return w + (h/4) * (k1 + 3 * k3)
#5.RK4
def rk4(t, w, h):
    k1 = h * f(t, w)
    k2 = h * f(t + h/2, w + k1/2)
    k3 = h * f(t + h/2, w + k2/2)
    k4 = h * f(t + h, w + k3)
    return w + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

# --- Multistep Method ---
def predictor_corrector_ab4_am3(t_vals, h, y0):
    n = len(t_vals)
    w = np.zeros(n)
    w[0] = y0
    
    # Generate starting values w1, w2, w3 using RK4
    for i in range(3):
        w[i+1] = rk4(t_vals[i], w[i], h)
        
    for i in range(3, n-1):
        t_i, t_im1, t_im2, t_im3 = t_vals[i], t_vals[i-1], t_vals[i-2], t_vals[i-3]
        f_i   = f(t_i, w[i])
        f_im1 = f(t_im1, w[i-1])
        f_im2 = f(t_im2, w[i-2])
        f_im3 = f(t_im3, w[i-3])
        
        # Predictor (Adams-Bashforth 4)
        w_pred = w[i] + (h/24) * (55*f_i - 59*f_im1 + 37*f_im2 - 9*f_im3)
        
        # Evaluator
        f_ip1 = f(t_vals[i+1], w_pred)
        
        # Corrector (Adams-Moulton 3)
        w[i+1] = w[i] + (h/24) * (9*f_ip1 + 19*f_i - 5*f_im1 + f_im2)
        
    return w

# --- Main Execution ---
a, b = 0, 5
y0 = 0.01
h = 0.4  # Test with step size 0.2 (can easily be changed to 0.1 or 0.05)

t_vals = np.arange(a, b + h, h)
exact_vals = exact_sol(t_vals)

results = {
    't': t_vals,
    'Exact': exact_vals,
    'Euler': np.zeros_like(t_vals),
    'Taylor2': np.zeros_like(t_vals),
    'Midpoint': np.zeros_like(t_vals),
    'Mod_Euler': np.zeros_like(t_vals),
    'Heun3': np.zeros_like(t_vals),
    'RK4': np.zeros_like(t_vals)
}

# Initialize w0
for key in results.keys():
    if key not in ['t', 'Exact']:
        results[key][0] = y0

# Compute One-Step Methods
for i in range(len(t_vals) - 1):
    t_i = t_vals[i]
    results['Euler'][i+1] = euler(t_i, results['Euler'][i], h)
    results['Taylor2'][i+1] = taylor_order2(t_i, results['Taylor2'][i], h)
    results['Midpoint'][i+1] = midpoint(t_i, results['Midpoint'][i], h)
    results['Mod_Euler'][i+1] = modified_euler(t_i, results['Mod_Euler'][i], h)
    results['Heun3'][i+1] = heun3(t_i, results['Heun3'][i], h)
    results['RK4'][i+1] = rk4(t_i, results['RK4'][i], h)

# Compute Predictor-Corrector
results['Pred_Corr'] = predictor_corrector_ab4_am3(t_vals, h, y0)

# Display Results DataFrame
df_results = pd.DataFrame(results)
print("Approximations (h = 0.2):")
print(df_results)

# Compute Absolute Errors
df_errors = pd.DataFrame({'t': t_vals})
for col in df_results.columns:
    if col not in ['t', 'Exact']:
        df_errors[col] = np.abs(df_results['Exact'] - df_results[col])

print("\nAbsolute Errors (h = 0.2):")
print(df_errors)


# plotting separately without Adams–Bashforth and Adams–Moulton
fig, axs = plt.subplots(2, 4, figsize=(15, 8)) # Creates a 2x4 grid
methods = ['Euler', 'Taylor2', 'Midpoint', 'Mod_Euler', 'Heun3', 'RK4', 'Pred_Corr']

# Loop through to plot each method on its own subplot
for i, method in enumerate(methods):
    row, col = i // 4, i % 4
    axs[row, col].plot(t_vals, exact_vals, 'k--', label='Exact')
    axs[row, col].plot(t_vals, results[method], label=method)
    axs[row, col].set_title(method)
    axs[row, col].legend()

plt.tight_layout()
plt.show()

# Plot Error Curves
plt.figure(figsize=(10, 6))
for col in df_errors.columns:
    if col != 't':
        plt.plot(df_errors['t'], df_errors[col], label=col)

plt.title('Absolute Errors for Problem A (h=0.4)')
plt.yscale('log') # Log scale is often best for visualizing drastically different error magnitudes
plt.xlabel('t')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(True)
plt.show()