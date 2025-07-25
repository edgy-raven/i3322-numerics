import os

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal


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

    def optimize(self, tol, force_symmetry=False):
        self.step_l()
        prev_value = self.value()
        while True:
            self.step_c()
            if force_symmetry and not np.allclose(self.c + self.c[::-1], 0):
                self.c = 0.5 * (self.c - self.c[::-1])
            self.step_l()
            if force_symmetry and not np.allclose(self.l - self.l[::-1], 0):
                self.l = 0.5 * (self.l + self.l[::-1])
                self.l = self.l / np.linalg.norm(self.l)

            new_value = self.value()
            if abs(new_value - prev_value) < tol:
                return self
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

    def extend(self):
        min_ix = np.argmax(np.diff(self.c[: self.n // 2]))
        return Solution(
            n=self.n + 2,
            l=None,
            c=np.concat(
                [
                    self.c[:min_ix],
                    [self.c[min_ix]],
                    self.c[min_ix:-min_ix],
                    [self.c[-min_ix]],
                    self.c[-min_ix:],
                ]
            ),
        )

    def write(self, file_path):
        with open(file_path, "w") as f:
            f.write(f"c = [{", ".join(map(str, self.c))}]'\n")
            f.write(f"l = [{", ".join(map(str, self.l))}];\n")
            f.write(f"v = {self.value()};\n")


def make_initial_c(n):
    return np.array([1] + ([0.9] * (n // 2)) + ([-0.9] * (n // 2)) + [-1])


def main(start, end, tol, force_symmetry, warm_start, write_path):
    soln = Solution(n=start, l=None, c=make_initial_c(start)).optimize(
        tol, force_symmetry
    )
    prev_value = soln.value()

    while end is None or soln.n <= end:
        if write_path:
            soln.write(os.path.join(write_path, f"{soln.n}.m"))
        if warm_start:
            soln = soln.extend()
        else:
            new_n = soln.n + 2
            soln = Solution(n=new_n, l=None, c=make_initial_c(new_n))

        soln.optimize(tol, force_symmetry)
        new_value = soln.value()
        if end is None and new_value <= prev_value:
            break
        prev_value = new_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=209)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--tol", type=float, default=1e-20)
    parser.add_argument("--force_symmetry", type=bool, default=False)
    parser.add_argument("--warm_start", type=bool, default=True)
    parser.add_argument("--write_path", type=str, default=None)
    args = parser.parse_args()

    main(**vars(args))
