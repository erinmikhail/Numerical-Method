import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sympy as sp

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from tests2_2 import get_system


# Настройки диапазонов и начальных точек под каждый вариант с учетом ОДЗ
VARIANT_CONFIGS = {
    # Группа 1 (1-3)
    1: {"x_bounds": (-4.0, 4.0), "y_bounds": (-2.0, 4.0), "x0": 1.5, "y0": 1.0, "note": "ОДЗ: Все R"},
    2: {"x_bounds": (-5.0, 5.0), "y_bounds": (-2.0, 5.0), "x0": 2.0, "y0": 1.0, "note": "ОДЗ: Все R"},
    3: {"x_bounds": (-6.0, 6.0), "y_bounds": (-2.0, 6.0), "x0": 2.5, "y0": 1.0, "note": "ОДЗ: Все R"},

    # Группа 2 (4-6)
    4: {"x_bounds": (-0.9, 4.0), "y_bounds": (-2.0, 4.0), "x0": 1.0, "y0": 1.0, "note": "ОДЗ: x1 > -1"},
    5: {"x_bounds": (-0.9, 4.0), "y_bounds": (-1.0, 5.0), "x0": 1.0, "y0": 2.0, "note": "ОДЗ: x1 > -1"},
    6: {"x_bounds": (-0.9, 4.0), "y_bounds": (0.0, 6.0), "x0": 1.0, "y0": 3.0, "note": "ОДЗ: x1 > -1"},

    # Группа 3 (7-9)
    7: {"x_bounds": (-3.0, 3.0), "y_bounds": (-3.0, 3.0), "x0": -1.0, "y0": 0.5, "note": "ОДЗ: Все R"},
    8: {"x_bounds": (-4.0, 4.0), "y_bounds": (-4.0, 4.0), "x0": -2.0, "y0": 0.5, "note": "ОДЗ: Все R"},
    9: {"x_bounds": (-5.0, 5.0), "y_bounds": (-5.0, 5.0), "x0": -3.0, "y0": 0.5, "note": "ОДЗ: Все R"},

    # Группа 4 (10-12)
    10: {"x_bounds": (-3.0, 3.0), "y_bounds": (-3.0, 3.0), "x0": 1.5, "y0": 1.5, "note": "ОДЗ: Все R"},
    11: {"x_bounds": (-2.0, 4.0), "y_bounds": (-2.0, 4.0), "x0": 2.5, "y0": 2.5, "note": "ОДЗ: Все R"},
    12: {"x_bounds": (-1.0, 5.0), "y_bounds": (-1.0, 5.0), "x0": 3.5, "y0": 3.5, "note": "ОДЗ: Все R"},

    # Группа 5 (13-15)
    13: {"x_bounds": (-2.5, 2.5), "y_bounds": (-1.5, 1.5), "x0": 1.0, "y0": 0.8, "note": "Эллипс"},
    14: {"x_bounds": (-3.5, 3.5), "y_bounds": (-2.0, 2.0), "x0": 1.5, "y0": 0.8, "note": "Эллипс"},
    15: {"x_bounds": (-4.5, 4.5), "y_bounds": (-2.5, 2.5), "x0": 2.0, "y0": 0.8, "note": "Эллипс"},

    # Группа 6 (16-18)
    16: {"x_bounds": (-1.5, 1.5), "y_bounds": (-1.5, 1.5), "x0": 0.4, "y0": 0.7, "note": "ОДЗ: Все R"},
    17: {"x_bounds": (-1.5, 1.5), "y_bounds": (-1.5, 1.5), "x0": 0.3, "y0": 0.4, "note": "ОДЗ: Все R"},
    18: {"x_bounds": (-1.5, 1.5), "y_bounds": (-1.5, 1.5), "x0": 0.2, "y0": 0.3, "note": "ОДЗ: Все R"},

    # Группа 7 (19-21)
    19: {"x_bounds": (-3.0, 3.0), "y_bounds": (0.05, 4.0), "x0": 1.5, "y0": 1.2, "note": "ОДЗ: x2 > 0!"},
    20: {"x_bounds": (-3.0, 3.0), "y_bounds": (0.05, 4.0), "x0": 1.5, "y0": 1.0, "note": "ОДЗ: x2 > 0!"},
    21: {"x_bounds": (-3.0, 3.0), "y_bounds": (0.05, 4.0), "x0": 1.5, "y0": 0.8, "note": "ОДЗ: x2 > 0!"},

    # Группа 8 (22-24)
    22: {"x_bounds": (-1.3, 1.3), "y_bounds": (-2.0, 2.0), "x0": 0.8, "y0": 1.0, "note": "ОДЗ: x1 != pi/2"},
    23: {"x_bounds": (-1.3, 1.3), "y_bounds": (-2.0, 2.0), "x0": 0.6, "y0": 0.7, "note": "ОДЗ: x1 != pi/2"},
    24: {"x_bounds": (-1.3, 1.3), "y_bounds": (-2.0, 2.0), "x0": 0.5, "y0": 0.5, "note": "ОДЗ: x1 != pi/2"},

    # Группа 9 (25-27)
    25: {"x_bounds": (-3.0, 3.0), "y_bounds": (-0.99, 3.0), "x0": 1.0, "y0": 1.0, "note": "ОДЗ: x2 >= -a"},
    26: {"x_bounds": (-3.0, 3.0), "y_bounds": (-1.99, 3.0), "x0": 1.0, "y0": 1.0, "note": "ОДЗ: x2 >= -a"},
    27: {"x_bounds": (-3.0, 3.0), "y_bounds": (-2.99, 3.0), "x0": 1.0, "y0": 1.0, "note": "ОДЗ: x2 >= -a"},

    # Группа 10 (28-30)
    28: {"x_bounds": (-1.0, 4.0), "y_bounds": (-1.0, 3.0), "x0": 1.3, "y0": 0.2, "note": "ОДЗ: Все R"},
    29: {"x_bounds": (-1.0, 5.0), "y_bounds": (-1.0, 3.0), "x0": 1.5, "y0": 0.2, "note": "ОДЗ: Все R"},
    30: {"x_bounds": (-1.0, 6.0), "y_bounds": (-1.0, 3.0), "x0": 1.8, "y0": 0.2, "note": "ОДЗ: Все R"},
}


class SystemSolverApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Лабораторная работа 2-2: Системы нелинейных уравнений")
        self.geometry("1180x800")

        self._create_widgets()
        self._on_variant_change()

    def _create_widgets(self):
        control_frame = ttk.LabelFrame(self, text=" Настройки и выбор варианта ", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Выбор варианта
        ttk.Label(control_frame, text="Выберите вариант (1-30):").pack(anchor=tk.W, pady=(0, 2))
        self.variant_var = tk.IntVar(value=1)
        variant_cb = ttk.Combobox(
            control_frame,
            textvariable=self.variant_var,
            values=list(range(1, 31)),
            state="readonly",
            width=15
        )
        variant_cb.pack(anchor=tk.W, pady=(0, 10))
        variant_cb.bind("<<ComboboxSelected>>", lambda e: self._on_variant_change())

        # Вывод уравнений и подсказок
        ttk.Label(control_frame, text="Система уравнений:").pack(anchor=tk.W)
        self.eq1_label = ttk.Label(control_frame, text="", font=("Courier", 9, "bold"), wraplength=260)
        self.eq1_label.pack(anchor=tk.W)
        self.eq2_label = ttk.Label(control_frame, text="", font=("Courier", 9, "bold"), wraplength=260)
        self.eq2_label.pack(anchor=tk.W, pady=(0, 5))

        self.note_label = ttk.Label(control_frame, text="", font=("Arial", 8, "italic"), foreground="blue")
        self.note_label.pack(anchor=tk.W, pady=(0, 10))

        # Настройки отображения графика
        ttk.Label(control_frame, text="Границы X [x_min, x_max]:").pack(anchor=tk.W)
        grid_x = ttk.Frame(control_frame)
        grid_x.pack(anchor=tk.W, pady=(0, 5))
        self.xmin_entry = ttk.Entry(grid_x, width=8)
        self.xmin_entry.grid(row=0, column=0)
        self.xmax_entry = ttk.Entry(grid_x, width=8)
        self.xmax_entry.grid(row=0, column=1, padx=5)

        ttk.Label(control_frame, text="Границы Y [y_min, y_max]:").pack(anchor=tk.W)
        grid_y = ttk.Frame(control_frame)
        grid_y.pack(anchor=tk.W, pady=(0, 10))
        self.ymin_entry = ttk.Entry(grid_y, width=8)
        self.ymin_entry.grid(row=0, column=0)
        self.ymax_entry = ttk.Entry(grid_y, width=8)
        self.ymax_entry.grid(row=0, column=1, padx=5)

        ttk.Button(control_frame, text=" Обновить график", command=self._plot_contours).pack(fill=tk.X, pady=(0, 15))

        # Начальное приближение
        ttk.Label(control_frame, text="Начальная точка (x1_0, x2_0):").pack(anchor=tk.W)
        grid_init = ttk.Frame(control_frame)
        grid_init.pack(anchor=tk.W, pady=(0, 10))
        ttk.Label(grid_init, text="x1:").grid(row=0, column=0)
        self.x0_entry = ttk.Entry(grid_init, width=8)
        self.x0_entry.grid(row=0, column=1, padx=(2, 8))
        ttk.Label(grid_init, text="x2:").grid(row=0, column=2)
        self.y0_entry = ttk.Entry(grid_init, width=8)
        self.y0_entry.grid(row=0, column=3)

        # Точность
        ttk.Label(control_frame, text="Точность (eps):").pack(anchor=tk.W)
        self.eps_entry = ttk.Entry(control_frame, width=15)
        self.eps_entry.insert(0, "1e-6")
        self.eps_entry.pack(anchor=tk.W, pady=(0, 15))

        # Кнопка решения
        ttk.Button(control_frame, text=" Найти решение", command=self._solve).pack(fill=tk.X, pady=(0, 15))

        # Результаты
        ttk.Label(control_frame, text="Результаты вычислений:").pack(anchor=tk.W)
        self.result_text = tk.Text(control_frame, width=34, height=13, font=("Consolas", 8))
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Правая панель с графиками
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax_sys = self.fig.add_subplot(211)
        self.ax_err = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

    def _on_variant_change(self):
        v = self.variant_var.get()
        _, _, sys_eqs = get_system(v)
        
        self.eq1_label.config(text=f"1) {sys_eqs[0]} = 0")
        self.eq2_label.config(text=f"2) {sys_eqs[1]} = 0")

        cfg = VARIANT_CONFIGS.get(v, {"x_bounds": (-3, 3), "y_bounds": (-3, 3), "x0": 1.0, "y0": 1.0, "note": ""})
        
        self.note_label.config(text=f"Подсказка: {cfg['note']}")

        # Обновление полей ввода пресетами
        self.xmin_entry.delete(0, tk.END); self.xmin_entry.insert(0, str(cfg["x_bounds"][0]))
        self.xmax_entry.delete(0, tk.END); self.xmax_entry.insert(0, str(cfg["x_bounds"][1]))
        self.ymin_entry.delete(0, tk.END); self.ymin_entry.insert(0, str(cfg["y_bounds"][0]))
        self.ymax_entry.delete(0, tk.END); self.ymax_entry.insert(0, str(cfg["y_bounds"][1]))

        self.x0_entry.delete(0, tk.END); self.x0_entry.insert(0, str(cfg["x0"]))
        self.y0_entry.delete(0, tk.END); self.y0_entry.insert(0, str(cfg["y0"]))

        self._plot_contours()

    def _get_lambdified(self):
        v = self.variant_var.get()
        x, y, sys_eqs = get_system(v)
        f1, f2 = sys_eqs[0], sys_eqs[1]

        J_matrix = sp.Matrix([f1, f2]).jacobian([x, y])

        f1_num = sp.lambdify((x, y), f1, 'numpy')
        f2_num = sp.lambdify((x, y), f2, 'numpy')
        J_num = sp.lambdify((x, y), J_matrix, 'numpy')

        return f1_num, f2_num, J_num

    def _plot_contours(self):
        try:
            xmin = float(self.xmin_entry.get())
            xmax = float(self.xmax_entry.get())
            ymin = float(self.ymin_entry.get())
            ymax = float(self.ymax_entry.get())

            f1, f2, _ = self._get_lambdified()

            X, Y = np.meshgrid(np.linspace(xmin, xmax, 300), np.linspace(ymin, ymax, 300))

            # Подавляем предупреждения NumPy при вычислении ОДЗ (логарифмы/корни)
            with np.errstate(invalid='ignore', divide='ignore'):
                Z1 = f1(X, Y)
                Z2 = f2(X, Y)

            self.ax_sys.clear()
            cs1 = self.ax_sys.contour(X, Y, Z1, levels=[0], colors='red')
            cs2 = self.ax_sys.contour(X, Y, Z2, levels=[0], colors='blue')

            if len(cs1.collections) > 0: cs1.collections[0].set_label('f1(x1, x2) = 0')
            if len(cs2.collections) > 0: cs2.collections[0].set_label('f2(x1, x2) = 0')

            self.ax_sys.grid(True, linestyle=':', alpha=0.6)
            self.ax_sys.set_title("Графическое отделение корней")
            self.ax_sys.set_xlabel("x1")
            self.ax_sys.set_ylabel("x2")
            self.ax_sys.legend(loc='upper right')
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось построить график: {e}")

    def _newton_system(self, f1, f2, J_func, x0, y0, eps, max_iter=100):
        x_curr, y_curr = x0, y0
        errors = []
        path = [(x_curr, y_curr)]

        for k in range(1, max_iter + 1):
            F = np.array([f1(x_curr, y_curr), f2(x_curr, y_curr)], dtype=float)
            J = np.array(J_func(x_curr, y_curr), dtype=float)

            if np.isnan(F).any() or np.isnan(J).any():
                raise ValueError("Точка вышла за пределы ОДЗ (появились NaN).")

            if abs(np.linalg.det(J)) < 1e-12:
                raise ValueError(f"Якобиан вырожден на итерации {k}.")

            delta = np.linalg.solve(J, -F)
            x_next, y_next = x_curr + delta[0], y_curr + delta[1]

            err = np.linalg.norm([x_next - x_curr, y_next - y_curr])
            errors.append(err)
            path.append((x_next, y_next))

            F_next = np.array([f1(x_next, y_next), f2(x_next, y_next)], dtype=float)
            if err < eps or np.linalg.norm(F_next) < eps:
                return x_next, y_next, k, errors, path

            x_curr, y_curr = x_next, y_next

        raise RuntimeError("Метод Ньютона не сошелся за отведенные итерации.")

    def _simple_iteration_system(self, f1, f2, J_func, x0, y0, eps, max_iter=500):
        J0 = np.array(J_func(x0, y0), dtype=float)
        if np.isnan(J0).any() or abs(np.linalg.det(J0)) < 1e-12:
            raise ValueError("Якобиан J(x0, y0) вырожден или вне ОДЗ. Измените начальную точку.")

        Lambda = np.linalg.inv(J0)

        x_curr, y_curr = x0, y0
        errors = []
        path = [(x_curr, y_curr)]

        for k in range(1, max_iter + 1):
            F = np.array([f1(x_curr, y_curr), f2(x_curr, y_curr)], dtype=float)
            if np.isnan(F).any():
                raise ValueError("Точка вышла за пределы ОДЗ во время итераций.")

            delta = -Lambda @ F
            x_next, y_next = x_curr + delta[0], y_curr + delta[1]

            err = np.linalg.norm([x_next - x_curr, y_next - y_curr])
            errors.append(err)
            path.append((x_next, y_next))

            F_next = np.array([f1(x_next, y_next), f2(x_next, y_next)], dtype=float)
            if err < eps or np.linalg.norm(F_next) < eps:
                return x_next, y_next, k, errors, path

            x_curr, y_curr = x_next, y_next

        raise RuntimeError("Метод простой итерации не сошелся.")

    def _solve(self):
        try:
            x0 = float(self.x0_entry.get())
            y0 = float(self.y0_entry.get())
            eps = float(self.eps_entry.get())

            f1, f2, J_func = self._get_lambdified()

            xn, yn, kn, errs_n, path_n = self._newton_system(f1, f2, J_func, x0, y0, eps)
            xsi, ysi, ksi, errs_si, path_si = self._simple_iteration_system(f1, f2, J_func, x0, y0, eps)

            # Вывод текстового отчета
            self.result_text.delete(1.0, tk.END)
            res_str = (
                f"--- МЕТОД НЬЮТОНА ---\n"
                f"x1* = {xn:.6f}, x2* = {yn:.6f}\n"
                f"Итераций: {kn}\n"
                f"||F|| = {np.linalg.norm([f1(xn, yn), f2(xn, yn)]):.2e}\n\n"
                f"--- ПРОСТАЯ ИТЕРАЦИЯ ---\n"
                f"x1* = {xsi:.6f}, x2* = {ysi:.6f}\n"
                f"Итераций: {ksi}\n"
                f"||F|| = {np.linalg.norm([f1(xsi, ysi), f2(xsi, ysi)]):.2e}\n"
            )
            self.result_text.insert(tk.END, res_str)

            # Отображение решения на графике
            self._plot_contours()
            path_n_arr = np.array(path_n)
            self.ax_sys.plot(path_n_arr[:, 0], path_n_arr[:, 1], 'g--s', label='Траектория Ньютона', markersize=4)
            self.ax_sys.plot(xn, yn, 'ro', markersize=8, label=f'Корень ({xn:.3f}, {yn:.3f})')
            self.ax_sys.legend(loc='upper right')

            # График сходимости
            self.ax_err.clear()
            self.ax_err.plot(range(1, ksi + 1), errs_si, 'r-o', label=f'Простая итерация ({ksi} итер.)')
            self.ax_err.plot(range(1, kn + 1), errs_n, 'g-s', label=f'Ньютон ({kn} итер.)')
            self.ax_err.set_yscale('log')
            self.ax_err.grid(True, which="both", linestyle=':', alpha=0.6)
            self.ax_err.set_title("Норма погрешности ||x_{k} - x_{k-1}|| (логарифм)")
            self.ax_err.set_xlabel("Номер итерации (k)")
            self.ax_err.set_ylabel("Погрешность")
            self.ax_err.legend()

            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Ошибка вычислений", str(e))


if __name__ == "__main__":
    app = SystemSolverApp()
    app.mainloop()