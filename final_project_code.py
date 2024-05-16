import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import kron, eye
import time

# Helper Functions
def circular_index(array, idx):
    return idx % len(array)


def f_prime(c, a, b):
    return a * c + b * c ** 3


def f(c, a, b):
    return a * c ** 2 / 2 + b * c ** 4 / 4


def boundary_condition(x):
    # Periodic Boundary Condition with 2 ghost cells
    N = x.shape[0]
    for i in range(N):
        x[0, i] = x[2, i]
        x[1, i] = x[2, i]
        x[N - 1, i] = x[N - 3, i]
        x[N - 2, i] = x[N - 3, i]
        x[i, 0] = x[i, 2]
        x[i, 1] = x[i, 2]
        x[i, N - 1] = x[i, N - 3]
        x[i, N - 2] = x[i, N - 3]


def total_energy(c, a, b, kappa):
    bulk_energy = np.sum(f(c, a, b))
    dcdx, dcdy = np.gradient(c)  # 2nd order central difference
    grad_c = dcdx + dcdy
    interface_energy = kappa * np.sum(grad_c ** 2)
    return bulk_energy + interface_energy


def discrete_laplacians(c_old, i, j, h, a, b):
    # Defining components
    cij = c_old[i, j]
    ci1j = c_old[i + 1, j]
    cij1 = c_old[i, j + 1]
    ci_1j = c_old[i - 1, j]
    cij_1 = c_old[i, j - 1]
    ci2j = c_old[i + 2, j]
    cij2 = c_old[i, j + 2]
    ci_2j = c_old[i - 2, j]
    cij_2 = c_old[i, j - 2]

    # Central difference scheme for nonlinear 2nd derivative
    laplacian_fprime_c = (f_prime(ci1j, a, b) + f_prime(ci_1j, a, b) + f_prime(cij1, a, b) + f_prime(cij_1, a,
                                                                                                     b) - 4 * f_prime(
        cij, a, b)) / h ** 2

    # Central difference scheme for 4th derivative
    laplacian2_c = (cij2 + cij_2 - 4 * ci_1j - 4 * ci1j + ci2j + ci_2j + - 4 * cij_1 - 4 * cij1 + 12 * cij) / h ** 4
    return laplacian2_c, laplacian_fprime_c


def difference_matrix_2d(n, h):
    I = eye(n)
    I1 = eye(n, k=1, )
    I_1 = eye(n, k=-1)
    D = (-2 * I + I1 + I_1)
    return (1 / h ** 2) * D, I


def laplacian_2D(nx, ny, hx, hy):
    Dx, Ix = difference_matrix_2d(nx, hx)
    Dy, Iy = difference_matrix_2d(ny, hy)

    return kron(Dx, Iy) + kron(Ix, Dy)


def raphson_newton(t_final, c0, N, M, a, b, kappa, nitertot=100, plot=False):
    # Define the grid with two ghost cells on each edge
    h = 1 / N
    dt = 0.8 * (h ** 4 / (8 * kappa * M))

    print("Number of Cells =", N)

    c = c0.copy()
    boundary_condition(c)

    energy_vector = [total_energy(c[2:-2, 2:-2], a, b, kappa)]
    time_vector = [0]

    niter = 0
    t = 0
    compute_time = 0

    while niter < nitertot and t < t_final:
        start = time.time()
        laplacian = laplacian_2D(N, N, h, h)
        Jacobian = -M * kappa * laplacian * laplacian + M * a * laplacian + (3 * b + a) * M * laplacian

        f = np.zeros((N, N))
        for i in range(2, N + 2):
            for j in range(2, N + 2):
                laplacian2_c, laplacian_fprime_c = discrete_laplacians(c, i, j, h, a, b)
                f[i - 2, j - 2] = M * laplacian_fprime_c - kappa * M * laplacian2_c
        f = f.flatten()

        df = np.linalg.solve(kron(eye(N), eye(N)).toarray() - Jacobian * dt, f * dt)
        c[2:-2, 2:-2] = c[2:-2, 2:-2] + df.reshape(N, N)
        boundary_condition(c)

        epsilon = np.max(abs(df))
        time_vector.append(t + dt)
        if epsilon < 0.1:
            dt = dt * 2
        t = t + dt  # update time and iteration counter
        niter += 1
        energy_vector.append(total_energy(c, a, b, kappa))
        end = time.time()
        compute_time += end - start
        if plot:
            if niter <= 100:
                if niter % 10 == 0:
                    plt.imshow(c[2:-2, 2:-2], cmap='hot')
                    plt.title(f"Concentration Iteration {niter} at time = {t:.3e}")
                    plt.colorbar()
                    plt.savefig(f"raphson_newton_{niter}.png")
                    plt.show()
            elif niter <= 1000:
                if niter % 250 == 0:
                    plt.imshow(c[2:-2, 2:-2], cmap='hot')
                    plt.title(f"Concentration Iteration {niter} at time = {t:.3e}")
                    plt.colorbar()
                    plt.savefig(f"raphson_newton_{niter}.png")
                    plt.show()
            elif niter > 1000:
                if niter % 2500 == 0:
                    plt.imshow(c[2:-2, 2:-2], cmap='hot')
                    plt.title(f"Concentration Iteration {niter} at time = {t:.3e}")
                    plt.colorbar()
                    plt.savefig(f"raphson_newton_{niter}.png")
                    plt.show()

    if niter == nitertot:
        print("Maximum number of iterations reached!")
    elif t > t_final:
        print(f"Reached final time in {niter} iterations")

    if plot:
        plt.scatter(time_vector, energy_vector)
        plt.plot(time_vector, energy_vector)
        plt.xscale('log')
        plt.xlabel("Time (s)")
        plt.ylabel("Total Energy")
        plt.title("Total Energy vs Time - Raphson Newton")
        plt.show()

    return c, time_vector, energy_vector, compute_time


def RK4(t_final, c0, N, M, a, b, kappa, nitertot=None, plot=False):
    """
    Function to solve the Cahn-Hilliard equation in 2D using a second order finite difference scheme for second order derivative and forward euler temporal discretization.
    """
    # Define the grid
    h = 1 / N

    # Set run parameters to ensure convergence. Ensures the Courant number is less than 1/8 (limiting factor for stability)
    dt = 0.1 * (h ** 4 / (8 * kappa * M))
    if nitertot is None:
        nitertot = int(t_final / dt)
    print("Number of Cells =", N, " Number of Time Iterations=", nitertot)

    # init time and iteration counter
    t = 0
    niter = 0

    c = c0.copy()  # initial state
    boundary_condition(c)

    energy_vector = [total_energy(c, a, b, kappa)]
    time_vector = [0]
    compute_time = 0

    while (t < t_final and niter <= nitertot):
        start = time.time()
        c_old = c.copy()

        # Update c
        for i in range(2, N + 2):
            for j in range(2, N + 2):
                laplacian2_c, laplacian_fprime_c = discrete_laplacians(c_old, i, j, h, a, b)
                k1 = M * (laplacian_fprime_c) - kappa * M * (laplacian2_c)

                c2 = c_old + dt / 2 * k1
                laplacian2_c, laplacian_fprime_c = discrete_laplacians(c2, i, j, h, a, b)
                k2 = M * (laplacian_fprime_c) - kappa * M * (laplacian2_c)

                c3 = c_old + dt / 2 * k2
                laplacian2_c, laplacian_fprime_c = discrete_laplacians(c3, i, j, h, a, b)
                k3 = M * (laplacian_fprime_c) - kappa * M * (laplacian2_c)

                c4 = c_old + dt * k3
                laplacian2_c, laplacian_fprime_c = discrete_laplacians(c4, i, j, h, a, b)
                k4 = M * (laplacian_fprime_c) - kappa * M * (laplacian2_c)

                c[i, j] = c_old[i, j] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        boundary_condition(c)

        t = t + dt  # update time and iteration counter
        niter = niter + 1
        energy_vector.append(total_energy(c, a, b, kappa))
        time_vector.append(t)
        end = time.time()
        compute_time += end - start

        if plot:
            if niter % (nitertot / 10) == 0:
                plt.imshow(c[2:-2, 2:-2], cmap='hot')
                plt.title(f"Concentration Iteration {niter} at time = {t:.3e}")
                plt.colorbar()
                plt.savefig(f"RK4_{niter}.png")
                plt.show()
    if niter == nitertot:
        print("Maximum number of iterations reached!")
    elif t > t_final:
        print(f"Reached final time in {niter} iterations")

    if plot:
        plt.scatter(time_vector, energy_vector)
        plt.plot(time_vector, energy_vector)
        plt.xlabel("Time (s)")
        plt.xlabel("Time (s)")
        plt.ylabel("Total Energy")
        plt.title("Total Energy vs Time - RK4")
        plt.savefig("RK4_energy.png")
        plt.show()

    return c, time_vector, energy_vector, compute_time


def ESDIRK(t_final, c0, N, M, a, b, kappa, nitertot=None, plot=False):
    # Define the grid with two ghost cells on each edge
    h = 1 / N
    dt = 0.8 * (h ** 4 / (8 * kappa * M))

    print("Number of Cells =", N)

    c = c0.copy()
    boundary_condition(c)

    energy_vector = [total_energy(c[2:-2, 2:-2], a, b, kappa)]
    time_vector = [0]

    niter = 0
    t = 0
    compute_time = 0

    # ESDIRK coefficients
    alpha = [[0.25, 0.25],
             [8611 / 62500, -1743 / 31250, 0.25],
             [5012029 / 34652500, -654441 / 2922500, 174375 / 388108, 0.25],
             [15267082809 / 155376265600, -71443401 / 120774400, 730878875 / 902184768, 2285395 / 8070912, 0.25],
             [82889 / 524892, 0, 15625 / 83664, 69875 / 102672, -2260 / 8211, 0.25]]
    beta = [82889 / 524892, 0, 15625 / 83664, 69875 / 102672, -2260 / 8211, 0.25]
    lam = 0.25
    while (t < t_final and niter <= nitertot):
        start = time.time()
        c_old = c.copy()

        # Define common Jacobian
        laplacian = laplacian_2D(N, N, h, h)
        Jacobian = -M * kappa * laplacian * laplacian + M * a * laplacian + (3 * b + a) * M * laplacian

        # Update c
        for i in range(2, N + 2):
            for j in range(2, N + 2):
                # Stage 1
                c1 = c_old.copy()
                laplacian2_c, laplacian_fprime_c = discrete_laplacians(c1, i, j, h, a, b)
                k1 = M * laplacian_fprime_c - kappa * M * laplacian2_c

                # Stage 2
                c2 = c1 + alpha[0][0] * dt * k1
                f = np.zeros((N, N))
                for i in range(2, N + 2):
                    for j in range(2, N + 2):
                        laplacian2_c, laplacian_fprime_c = discrete_laplacians(c2, i, j, h, a, b)
                        f[i - 2, j - 2] = M * laplacian_fprime_c - kappa * M * laplacian2_c
                f = f.flatten()
                df = np.linalg.solve(kron(eye(N), eye(N)).toarray() - Jacobian * dt, lam * f * dt)
                c2[2:-2, 2:-2] = c2[2:-2, 2:-2] + df.reshape(N, N)
                boundary_condition(c2)
                laplacian2_c, laplacian_fprime_c = discrete_laplacians(c2, i, j, h, a, b)
                k2 = M * laplacian_fprime_c - kappa * M * laplacian2_c

                # Stage 3
                c3 = c1 + alpha[1][0] * dt * k1 + alpha[1][1] * dt * k2
                f = np.zeros((N, N))
                for i in range(2, N + 2):
                    for j in range(2, N + 2):
                        laplacian2_c, laplacian_fprime_c = discrete_laplacians(c3, i, j, h, a, b)
                        f[i - 2, j - 2] = M * laplacian_fprime_c - kappa * M * laplacian2_c
                f = f.flatten()
                df = np.linalg.solve(kron(eye(N), eye(N)).toarray() - Jacobian * dt, lam * f * dt)
                c3[2:-2, 2:-2] = c3[2:-2, 2:-2] + df.reshape(N, N)
                boundary_condition(c3)
                laplacian2_c, laplacian_fprime_c = discrete_laplacians(c3, i, j, h, a, b)
                k3 = M * laplacian_fprime_c - kappa * M * laplacian2_c

                # Stage 4
                c4 = c1 + alpha[2][0] * dt * k1 + alpha[2][1] * dt * k2 + alpha[2][2] * dt * k3
                f = np.zeros((N, N))
                for i in range(2, N + 2):
                    for j in range(2, N + 2):
                        laplacian2_c, laplacian_fprime_c = discrete_laplacians(c4, i, j, h, a, b)
                        f[i - 2, j - 2] = M * laplacian_fprime_c - kappa * M * laplacian2_c
                f = f.flatten()
                df = np.linalg.solve(kron(eye(N), eye(N)).toarray() - Jacobian * dt, lam * f * dt)
                c4[2:-2, 2:-2] = c4[2:-2, 2:-2] + df.reshape(N, N)
                boundary_condition(c4)
                laplacian2_c, laplacian_fprime_c = discrete_laplacians(c4, i, j, h, a, b)
                k4 = M * laplacian_fprime_c - kappa * M * laplacian2_c

                # Stage 5
                c5 = c1 + alpha[3][0] * dt * k1 + alpha[3][1] * dt * k2 + alpha[3][2] * dt * k3 + alpha[3][3] * dt * k4
                f = np.zeros((N, N))
                for i in range(2, N + 2):
                    for j in range(2, N + 2):
                        laplacian2_c, laplacian_fprime_c = discrete_laplacians(c5, i, j, h, a, b)
                        f[i - 2, j - 2] = M * laplacian_fprime_c - kappa * M * laplacian2_c
                f = f.flatten()
                df = np.linalg.solve(kron(eye(N), eye(N)).toarray() - Jacobian * dt, lam * f * dt)
                c5[2:-2, 2:-2] = c5[2:-2, 2:-2] + df.reshape(N, N)
                boundary_condition(c5)
                laplacian2_c, laplacian_fprime_c = discrete_laplacians(c5, i, j, h, a, b)
                k5 = M * laplacian_fprime_c - kappa * M * laplacian2_c

                # Stage 6
                c6 = c1 + alpha[4][0] * dt * k1 + alpha[4][1] * dt * k2 + alpha[4][2] * dt * k3 + alpha[4][
                    3] * dt * k4 + alpha[4][4] * dt * k5
                f = np.zeros((N, N))
                for i in range(2, N + 2):
                    for j in range(2, N + 2):
                        laplacian2_c, laplacian_fprime_c = discrete_laplacians(c6, i, j, h, a, b)
                        f[i - 2, j - 2] = M * laplacian_fprime_c - kappa * M * laplacian2_c
                f = f.flatten()
                df = np.linalg.solve(kron(eye(N), eye(N)).toarray() - Jacobian * dt, lam * f * dt)
                c6[2:-2, 2:-2] = c6[2:-2, 2:-2] + df.reshape(N, N)
                boundary_condition(c6)
                laplacian2_c, laplacian_fprime_c = discrete_laplacians(c6, i, j, h, a, b)
                k6 = M * laplacian_fprime_c - kappa * M * laplacian2_c

                c[i, j] = c_old[i, j] + beta[0] * dt * k1 + beta[1] * dt * k2 + beta[2] * dt * k3 + beta[3] * dt * k4 + \
                          beta[4] * dt * k5 + beta[5] * dt * k6
        boundary_condition(c)

        epsilon = np.max(abs(c - c_old))
        time_vector.append(t + dt)
        if epsilon < 0.1:
            dt = dt * 2

        t = t + dt  # update time and iteration counter
        niter = niter + 1
        energy_vector.append(total_energy(c, a, b, kappa))
        time_vector.append(t)
        end = time.time()
        compute_time += end - start

        if plot:
            if niter <= 100:
                if niter % 10 == 0:
                    plt.imshow(c[2:-2, 2:-2], cmap='hot')
                    plt.title(f"Concentration Iteration {niter} at time = {t:.3e}")
                    plt.colorbar()
                    plt.savefig(f"ESDIRK_{niter}.png")
                    plt.show()
            elif niter < 1000:
                if niter % 250 == 0:
                    plt.imshow(c[2:-2, 2:-2], cmap='hot')
                    plt.title(f"Concentration Iteration {niter} at time = {t:.3e}")
                    plt.colorbar()
                    plt.savefig(f"ESDIRK_{niter}.png")
                    plt.show()
            elif niter > 1000:
                if niter % 2500 == 0:
                    plt.imshow(c[2:-2, 2:-2], cmap='hot')
                    plt.title(f"Concentration Iteration {niter} at time = {t:.3e}")
                    plt.colorbar()
                    plt.savefig(f"EDISRK_{niter}.png")
                    plt.show()

    if niter == nitertot:
        print("Maximum number of iterations reached!")
    elif t > t_final:
        print(f"Reached final time in {niter} iterations")

    if plot:
        plt.scatter(time_vector, energy_vector)
        plt.plot(time_vector, energy_vector)
        plt.xscale('log')
        plt.xlabel("Time (s)")
        plt.ylabel("Total Energy")
        plt.title("Total Energy vs Time - ESDIRk")
        plt.show()

    return c, time_vector, energy_vector, compute_time


def CH_solver_fourth_deriv(t_final, c0, N, M, a, b, kappa, nitertot=None, plot=False):
    """
    Function to solve the Cahn-Hilliard equation in 2D using a second order finite difference scheme for forth order derivative and forward euler temporal discretization.

    """
    # Define the grid

    h = 1 / N

    # Set run parameters to ensure convergence. Ensures the Courant number is less than 1/8 (limiting factor for stability)
    dt = 0.1 * (h ** 4 / (8 * kappa * M))
    if nitertot is None:
        nitertot = int(t_final / dt)
    print("Number of Cells =", N, " Number of Time Iterations=", nitertot)

    # Modified Courant number
    C = kappa * M * dt / h ** 4
    C_nonlinear = M * dt / h ** 2
    print(f"Modified Courant number = {C:.4f} is 10% of the maximum value for stability")

    # init time and iteration counter
    t = 0
    niter = 0

    c = c0.copy()  # initial state
    energy_vector = [total_energy(c, a, b, kappa)]
    time_vector = [0]
    compute_time = 0

    while (t < t_final and niter <= nitertot):
        start_time = time.time()
        c_old = c.copy()

        # Update c
        for i in range(2, N + 2):
            for j in range(2, N + 2):
                # Defining components
                cij = c_old[i, j]
                ci1j = c_old[i + 1, j]
                cij1 = c_old[i, j + 1]
                ci_1j = c_old[i - 1, j]
                cij_1 = c_old[i, j - 1]
                ci2j = c_old[i + 2, j]
                cij2 = c_old[i, j + 2]
                ci_2j = c_old[i - 2, j]
                cij_2 = c_old[i, j - 2]

                # Central difference scheme for nonlinear 2nd derivative
                laplacian_fprime_c = (
                            f_prime(ci1j, a, b) + f_prime(ci_1j, a, b) + f_prime(cij1, a, b) + f_prime(cij_1, a, b) - 4 * f_prime(
                        cij, a, b))

                # Central difference scheme for 4th derivative
                laplacian2_c = (cij2 + cij_2 - 4 * ci_1j - 4 * ci1j + ci2j + ci_2j + - 4 * cij_1 - 4 * cij1 + 12 * cij)
                c[i, j] = c_old[i, j] + C_nonlinear * (laplacian_fprime_c) - C * (laplacian2_c)

        boundary_condition(c)

        t = t + dt  # update time and iteration counter
        niter = niter + 1
        energy_vector.append(total_energy(c, a, b, kappa))
        time_vector.append(t)
        endtime = time.time()
        compute_time += endtime - start_time
        if plot:
            if niter % (nitertot / 10) == 0:
                plt.imshow(c[2:-2, 2:-2], cmap='hot')
                plt.title(f"Concentration Iteration {niter} at time = {t:.3e}")
                plt.colorbar()
                plt.savefig(f"RK1_{niter}.png")
                plt.show()

    if niter == nitertot:
        print("Maximum number of iterations reached!")
    elif t > t_final:
        print(f"Reached final time in {niter} iterations")

    if plot:
        plt.scatter(time_vector, energy_vector)
        plt.plot(time_vector, energy_vector)
        plt.xlabel("Time (s)")
        plt.xlabel("Time (s)")
        plt.ylabel("Total Energy")
        plt.title("Total Energy vs Time - RK1")
        plt.savefig("RK1_energy.png")
        plt.show()

    return c, time_vector, energy_vector, compute_time
