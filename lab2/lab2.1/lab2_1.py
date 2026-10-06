import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sympy as sp

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from tests_2_1 import get_equations


class SolverApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Лабораторная работа 2-1: Решение нелинейных уравнений")
        self.geometry("1100x750")

        # Загрузка уравнений из файла tests_2_1.py
        self.x_sym, self.equations = get_equations()

        self._create_widgets()
        self._on_variant_change()

    def _create_widgets(self):
        # Левая панель - настройки и ввод
        control_frame = ttk.LabelFrame(self, text=" Параметры и ввод ", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Выбор варианта
        ttk.Label(control_frame, text="Выберите вариант (1-30):").pack(anchor=tk.W, pady=(0, 2))
        self.variant_var = tk.IntVar(value=1)
        variant_cb = ttk.Combobox(
            control_frame, 
            textvariable=self.variant_var, 
            values=list(self.equations.keys()), 
            state="readonly",
            width=15
        )
        variant_cb.pack(anchor=tk.W, pady=(0, 10))
        variant_cb.bind("<<ComboboxSelected>>", lambda e: self._on_variant_change())

        # Отображение выбранного выражения
        ttk.Label(control_frame, text="Уравнение f(x) = 0:").pack(anchor=tk.W)
        self.eq_str_label = ttk.Label(control_frame, text="", font=("Courier", 10, "bold"), wraplength=250)
        self.eq_str_label.pack(anchor=tk.W, pady=(0, 15))

        # Границы интервала для построения графика
        ttk.Label(control_frame, text="Интервал для графика [a, b]:").pack(anchor=tk.W)
        grid_bounds = ttk.Frame(control_frame)
        grid_bounds.pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(grid_bounds, text="a:").grid(row=0, column=0)
        self.a_entry = ttk.Entry(grid_bounds, width=7)
        self.a_entry.insert(0, "0.01")
        self.a_entry.grid(row=0, column=1, padx=(2, 10))

        ttk.Label(grid_bounds, text="b:").grid(row=0, column=2)
        self.b_entry = ttk.Entry(grid_bounds, width=7)
        self.b_entry.insert(0, "3.0")
        self.b_entry.grid(row=0, column=3, padx=2)

        ttk.Button(control_frame, text=" Обновить график f(x)", command=self._plot_function).pack(fill=tk.X, pady=(0, 15))

        # Начальное приближение x0 и точность eps
        ttk.Label(control_frame, text="Начальное приближение x0 (> 0):").pack(anchor=tk.W)
        self.x0_entry = ttk.Entry(control_frame, width=15)
        self.x0_entry.insert(0, "1.0")
        self.x0_entry.pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(control_frame, text="Точность вычислений (eps):").pack(anchor=tk.W)
        self.eps_entry = ttk.Entry(control_frame, width=15)
        self.eps_entry.insert(0, "1e-6")
        self.eps_entry.pack(anchor=tk.W, pady=(0, 15))

        # Кнопка решения
        ttk.Button(control_frame, text=" Найти корень и проанализировать", command=self._solve).pack(fill=tk.X, pady=(0, 15))

        # Текстовое поле вывода результатов
        ttk.Label(control_frame, text="Результаты вычислений:").pack(anchor=tk.W)
        self.result_text = tk.Text(control_frame, width=32, height=14, font=("Consolas", 9))
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Правая панель - графики Matplotlib
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax_func = self.fig.add_subplot(211)
        self.ax_err = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

    def _on_variant_change(self):
        v = self.variant_var.get()
        eq = self.equations[v]
        self.eq_str_label.config(text=f"{eq} = 0")
        self._plot_function()

    def _get_lambdas(self):
        v = self.variant_var.get()
        eq = self.equations[v]
        diff_eq = sp.diff(eq, self.x_sym)

        f = sp.lambdify(self.x_sym, eq, 'numpy')
        df = sp.lambdify(self.x_sym, diff_eq, 'numpy')
        return f, df

    def _plot_function(self):
        try:
            a = float(self.a_entry.get())
            b = float(self.b_entry.get())
            f, _ = self._get_lambdas()

            x_vals = np.linspace(a, b, 500)
            y_vals = f(x_vals)

            self.ax_func.clear()
            self.ax_func.plot(x_vals, y_vals, 'b-', label='f(x)')
            self.ax_func.axhline(0, color='black', linewidth=0.8, linestyle='--')
            self.ax_func.grid(True, linestyle=':', alpha=0.6)
            self.ax_func.set_title("Графическое определение корня f(x) = 0")
            self.ax_func.set_xlabel("x")
            self.ax_func.set_ylabel("f(x)")
            self.ax_func.legend()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка построения графика: {e}")

    def _simple_iteration_method(self, f, df, x0, eps, max_iter=1000):
        """
        Метод простой итерации с автоматической релаксацией lambda = 1 / f'(x0)
        для обеспечения сходимости (|phi'(x)| < 1).
        """
        df0 = df(x0)
        if abs(df0) < 1e-12:
            raise ValueError("Производная f'(x0) близка к нулю. Выберите другое x0.")

        # Коэффициент релаксации gamma
        gamma = 1.0 / df0
        
        x_curr = x0
        errors = []

        for k in range(1, max_iter + 1):
            x_next = x_curr - gamma * f(x_curr)
            err = abs(x_next - x_curr)
            errors.append(err)

            if err < eps or abs(f(x_next)) < eps:
                return x_next, k, errors

            x_curr = x_next

        raise RuntimeError("Метод простой итерации не сошелся за отведенное число шагов.")

    def _newton_method(self, f, df, x0, eps, max_iter=100):
        """
        Метод Ньютона (касательных).
        """
        x_curr = x0
        errors = []

        for k in range(1, max_iter + 1):
            df_val = df(x_curr)
            if abs(df_val) < 1e-12:
                raise ValueError(f"Производная f'(x) близка к 0 на итерации {k}.")

            x_next = x_curr - f(x_curr) / df_val
            err = abs(x_next - x_curr)
            errors.append(err)

            if err < eps or abs(f(x_next)) < eps:
                return x_next, k, errors

            x_curr = x_next

        raise RuntimeError("Метод Ньютона не сошелся за отведенное число шагов.")

    def _solve(self):
        try:
            x0 = float(self.x0_entry.get())
            eps = float(self.eps_entry.get())
            if x0 <= 0:
                messagebox.showwarning("Предупреждение", "Корень должен быть положительным (x0 > 0).")

            f, df = self._get_lambdas()

            # Вычисление двумя методами
            root_si, k_si, errs_si = self._simple_iteration_method(f, df, x0, eps)
            root_n, k_n, errs_n = self._newton_method(f, df, x0, eps)

            # Вывод текста
            self.result_text.delete(1.0, tk.END)
            res_str = (
                f"--- МЕТОД НЬЮТОНА ---\n"
                f"Корень: {root_n:.8f}\n"
                f"Итераций: {k_n}\n"
                f"f(x): {f(root_n):.2e}\n\n"
                f"--- ПРОСТАЯ ИТЕРАЦИЯ ---\n"
                f"Корень: {root_si:.8f}\n"
                f"Итераций: {k_si}\n"
                f"f(x): {f(root_si):.2e}\n"
            )
            self.result_text.insert(tk.END, res_str)

            # График погрешностей
            self.ax_err.clear()
            self.ax_err.plot(range(1, k_si + 1), errs_si, 'r-o', label=f'Простая итерация ({k_si} шаг.)')
            self.ax_err.plot(range(1, k_n + 1), errs_n, 'g-s', label=f'Ньютон ({k_n} шаг.)')
            self.ax_err.set_yscale('log')
            self.ax_err.grid(True, which="both", linestyle=':', alpha=0.6)
            self.ax_err.set_title("Зависимость погрешности |x_{k} - x_{k-1}| от итерации (логарифм)")
            self.ax_err.set_xlabel("Номер итерации (k)")
            self.ax_err.set_ylabel("Погрешность")
            self.ax_err.legend()

            # Отметка корня на графике функции
            self._plot_function()
            self.ax_func.plot(root_n, f(root_n), 'ro', markersize=8, label=f'Корень ≈ {root_n:.4f}')
            self.ax_func.legend()

            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Ошибка вычислений", str(e))


if __name__ == "__main__":
    app = SolverApp()
    app.mainloop()