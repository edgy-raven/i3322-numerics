from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal

from matplotlib import pyplot as plt


@dataclass
class Solution:
    n: int
    l: np.ndarray  # will have n elements
    c: np.ndarray  # will have n + 1 number of elements, c_0 = 1, c_n = -1

    def value(self):
        c_plus = self.c[1:]
        c = self.c[:-1]
        l_plus = self.l[1:]
        l = self.l[:-1]

        diag = np.sum((c * c_plus + 0.5 * (c - c_plus) - 1) * (self.l**2))
        off_diag = np.sum(np.sqrt(1 - self.c[1:-1] ** 2) * l * l_plus)
        return diag + off_diag

    def optimize(self, tol=1e-13):
        self.step_l()
        prev_value = self.value()
        while True:
            self.step_c()
            self.step_l()
            new_value = self.value()
            if (new_value - prev_value) < tol:
                return
            prev_value = new_value

    def step_l(self):
        c_plus = self.c[1:]
        c = self.c[:-1]

        # Pal-Vertesi equation 23
        diag = c * c_plus + 0.5 * (c - c_plus) - 1
        off_diag = 0.5 * np.sqrt(1 - self.c[1:-1] ** 2)
        self.l = eigh_tridiagonal(diag, off_diag)[1][:, -1]

    def step_c(self):
        l_plus = self.l[1:]
        l = self.l[:-1]

        # Pal-Vertesi equation 22
        tau = (1 + 2 * self.c[2:]) * l_plus**2 - (1 - 2 * self.c[:-2]) * l**2
        # Pal-Vertesi equation 21
        self.c[1:-1] = tau / np.sqrt(tau**2 + 4 * (l**2) * (l_plus**2))


def main():
    dim = 89
    c = np.array([1] + ([0.9] * (dim // 2)) + ([-0.9] * (dim // 2)) + [-1])
    soln = Solution(n=dim, l=None, c=c)
    soln.optimize()
    plt.plot(soln.l)
    plt.savefig("l.png")

    plt.plot(soln.c)
    plt.savefig("c.png")


if __name__ == "__main__":
    main()
