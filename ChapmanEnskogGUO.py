import sympy as sp

print("======================================================================")
print("   РАЗЛОЖЕНИЕ ЧЕПМЕНА-ЭНСКОГО: D2Q9 BGK + CSF")
print("======================================================================")

# 1. Символы
x, y, t = sp.symbols('x y t')
rho = sp.Function('rho')(x, y, t)
u = sp.Function('u')(x, y, t)
v = sp.Function('v')(x, y, t)
tau = sp.Symbol('tau')
Fx = sp.Symbol('Fx')
Fy = sp.Symbol('Fy')
dt_sym = sp.Symbol('Δt')
epsilon = sp.Symbol('epsilon')
cs2 = sp.Rational(1, 3)

# 2. Решетка D2Q9
c_x = [0, 1, 0, -1, 0, 1, -1, -1, 1]
c_y = [0, 0, 1, 0, -1, 1, 1, -1, -1]
weights = [
    sp.Rational(4, 9),
    sp.Rational(1, 9), sp.Rational(1, 9), sp.Rational(1, 9), sp.Rational(1, 9),
    sp.Rational(1, 36), sp.Rational(1, 36), sp.Rational(1, 36), sp.Rational(1, 36)
]
Q = 9

# 3. Равновесные функции f_eq
f_eq = []
for i in range(Q):
    u_g = u + (Fx * dt_sym) / (2 * rho)
    v_g = v + (Fy * dt_sym) / (2 * rho)

    cu = c_x[i]*u_g + c_y[i]*v_g
    term1 = cu / cs2
    term2 = (cu**2) / (2 * cs2**2)
    term3 = -(u_g**2 + v_g**2) / (2 * cs2)
    feq_i = weights[i] * rho * (1 + term1 + term2 + term3)
    f_eq.append(feq_i)


# Проверка моментов
sum_rho = sp.simplify(sum(f_eq))
sum_mom_x = sp.simplify(sum(f_eq[i] * c_x[i] for i in range(Q)))
sum_mom_y = sp.simplify(sum(f_eq[i] * c_y[i] for i in range(Q)))
print("Нулевой момент равновесной функции распредления:")
sp.pprint(sum_rho)
print("Первый момент равновесной функции распредления X:")
sp.pprint(sum_mom_x)
print("Первый момент равновесной функции распредления Y:")
sp.pprint(sum_mom_y)

# ------------------------------------------------------------------------
# ШАГ 2: МАСШТАБЫ И РАЗЛОЖЕНИЕ LHS
# ------------------------------------------------------------------------

# 1. Разложение f
f_0 = sp.Function('f^(0)')(x,y,t)
f_1 = sp.Function('f^(1)')(x,y,t)
f_2 = sp.Function('f^(2)')(x,y,t)

f_expansion = f_0 + epsilon * f_1 + epsilon**2 * f_2
print(f"\nРазложение f: ")
sp.pprint(sp.Eq(sp.Symbol("f"),f_expansion))

# 2. Производные (Мульти-масштаб)
dt1 = sp.Symbol('d/dt1')
dt2 = sp.Symbol('d/dt2')
dx1 = sp.Symbol('d/dx1')
dy1 = sp.Symbol('d/dy1')

# Конвективный оператор D1 = dt1 + c * grad1
c_x_sym = sp.Symbol('c_x')
c_y_sym = sp.Symbol('c_y')
D1 = dt1 + c_x_sym * dx1 + c_y_sym * dy1

# 3. Разложение Левой Части (LHS) уравнения Больцмана
# LHS = f(x+c*dt, t+dt) - f(x,t)

Dt_operator = epsilon * D1 + epsilon**2 * dt2

# Применяем к f = f0 + eps*f1
# LHS = (dt * Dt_operator + dt**2/2 * Dt_operator**2) * (f0 + eps*f1)

# 1. Оператор первого порядка L1 = dt * Dt_operator
L1_f = dt_sym * (Dt_operator * f_0 + epsilon * Dt_operator * f_1)

# 2. Оператор второго порядка L2 = dt**2/2 * Dt_operator**2
L2_f = (dt_sym**2 / 2) * (Dt_operator**2 * f_0)

LHS_full = sp.expand(L1_f + L2_f)

# Отбрасываем O(eps^3)

LHS_eps1 = LHS_full.coeff(epsilon, 1)
LHS_eps2 = LHS_full.coeff(epsilon, 2)

print("\nРазложение Уравнения Больцмана (LHS):")
print("\nЧлены порядка epsilon^1 (Euler):")
sp.pprint(LHS_eps1)

print("\nЧлены порядка epsilon^2 (Navier-Stokes):")
sp.pprint(LHS_eps2)


# ------------------------------------------------------------------------
# ШАГ 3: РАЗЛОЖЕНИЕ ПРАВОЙ ЧАСТИ (RHS): BGK + GUO FORCE
# ------------------------------------------------------------------------
print("\n--- ШАГ 3: RHS (BGK + Guo Force) ---")
# Omega_BGK = -1/tau * (f - f_eq)

# f_eq = f0
Omega_BGK = -1/tau * (epsilon * f_1 + epsilon**2 * f_2)
# Важно: нужно раскрыть скобки, иначе SymPy видит выражение как Mul(-1/tau, Add(...))
# и coeff может не найти epsilon внутри скобок.
Omega_BGK = sp.expand(Omega_BGK)

print("\nОператор столкновений (Omega_BGK):")
sp.pprint(Omega_BGK)

# 2. Сила Гуо (Guo Force Source Term)
# S_i = eps * S_1
S_1 = sp.Function('S^(1)')(x,y,t) # Главный член силы (O(eps))

Force_Term = dt_sym * epsilon * S_1
print("\nСиловой член (Guo Force):")
sp.pprint(Force_Term)

print("\n--- Итоговые уравнения по порядкам epsilon ---")
# Порядок eps^1:
RHS_eps1 = Omega_BGK.coeff(epsilon,1) + Force_Term.coeff(epsilon,1)
Eq_eps1 = sp.Eq(LHS_eps1, RHS_eps1)

print("\nУравнение порядка epsilon^1:")
sp.pprint(Eq_eps1)

# Порядок eps^2:
RHS_eps2 = Omega_BGK.coeff(epsilon,2)
Eq_eps2 = sp.Eq(LHS_eps2, RHS_eps2)

print("\nУравнение порядка epsilon^2:")
sp.pprint(Eq_eps2)

# ------------------------------------------------------------------------
# ШАГ 4: ВЫЧИСЛЕНИЕ МОМЕНТОВ И ПОЛУЧЕНИЕ УРАВНЕНИЙ (NAVIER-STOKES)
# ------------------------------------------------------------------------
print(" ШАГ 4: ПОЛУЧЕНИЕ МАКРОСКОПИЧЕСКИХ УРАВНЕНИЙ")

# Формируем список S_i для силы Гуо (полная формула)
# S_i = w_i * (1 - 1/(2tau)) * [ (c.F)/cs2 + (c.F)(c.u)/cs4 - (F.u)/cs2 ]
Guo_S_list = []
factor_tau = (1 - 1 / (2 * tau))

for i in range(Q):
    u_g = u + (Fx * dt_sym) / (2 * rho)
    v_g = v + (Fy * dt_sym) / (2 * rho)

    c_dot_F = c_x[i] * Fx + c_y[i] * Fy
    c_dot_u = c_x[i] * u_g + c_y[i] * v_g
    F_dot_u = Fx * u_g + Fy * v_g

    term1 = c_dot_F / cs2
    term2 = (c_dot_F * c_dot_u) / (cs2 ** 2)
    term3 = -F_dot_u / cs2

    S_i_full = weights[i] * factor_tau * (term1 + term2 + term3)
    Guo_S_list.append(S_i_full)

momentForce0 = sp.simplify(sum(Guo_S_list))
momentForce1x = sp.simplify(sum(c_x[i] * Guo_S_list[i] for i in range(Q)))
momentForce1y = sp.simplify(sum(c_y[i] * Guo_S_list[i] for i in range(Q)))
momentForce2xx = sp.simplify(sum(c_x[i] * c_x[i] * Guo_S_list[i] for i in range(Q)))
momentForce2xy = sp.simplify(sum(c_x[i] * c_y[i] * Guo_S_list[i] for i in range(Q)))
momentForce2yy = sp.simplify(sum(c_y[i] * c_y[i] * Guo_S_list[i] for i in range(Q)))

print("Моменты от силы Гуо:")
print("Нулевой момент:")
sp.pprint(momentForce0)
print("Первый момент X:")
sp.pprint(momentForce1x)
print("Первый момент Y:")
sp.pprint(momentForce1y)
print("Второй момент XX:")
sp.pprint(momentForce2xx)
print("Второй момент XY:")
sp.pprint(momentForce2xy)
print("Второй момент YY:")
sp.pprint(momentForce2yy)

# --- АНАЛИЗ ПОРЯДКА EPS^1 (EULER) ---
print("\n--- Анализ порядка epsilon^1 (Euler) ---")

# 1. Уравнение непрерывности (Момент 0 от Eq_eps1)

print("Выводим уравнение непрерывности (mass conservation)...")
# Собираем выражение LHS
LHS_mass_eps1 = 0
RHS_mass_eps1 = 0

# Суммируем LHS и RHS
for i in range(Q):
    term_l = LHS_eps1.subs({c_x_sym: c_x[i], c_y_sym: c_y[i], f_0: f_eq[i]})
    LHS_mass_eps1 += term_l

    # RHS: -1/tau * f_1 + dt * S_1
    term_r = RHS_eps1.subs({S_1: Guo_S_list[i]})
    RHS_mass_eps1 += term_r

# Обработка f_1 в RHS:
# Мы знаем физически, что sum(f_1) = 0. Но SymPy видит 9 * f_1.
# Поскольку f_1 - неизвестная функция неравновесия, мы не можем "вычислить" её сумму, не задав её явно.
# Однако мы МОЖЕМ явно использовать свойство sum(f_1) = 0.
# Заменим f_1 на 0 в итоговой сумме RHS, так как это сумма неравновесных частей.
RHS_mass_eps1 = RHS_mass_eps1.subs(f_1, 0)
# Если бы f_1 был вектором f_1_i, сумма была бы нулем. Так как тут f_1 скаляр-символ, удаляем его постфактум.

print("Уравнение непрерывности:")
lhs_simple = sp.simplify(LHS_mass_eps1 / dt_sym)
rhs_simple = sp.simplify(RHS_mass_eps1 / dt_sym)

sp.pprint(sp.Eq(lhs_simple, rhs_simple))

# Перезаписываем уравнение непрерывности в терминах скорости по Гуо (u_g, v_g)
u_g_sym = sp.Function('u_g')(x, y, t)
v_g_sym = sp.Function('v_g')(x, y, t)

subs_to_ug = {
    u: u_g_sym - Fx * dt_sym / (2 * rho),
    v: v_g_sym - Fy * dt_sym / (2 * rho),
}

lhs_mass_ug = sp.simplify(lhs_simple.subs(subs_to_ug))
rhs_mass_ug = sp.simplify(rhs_simple.subs(subs_to_ug))

print("\nУравнение непрерывности в терминах u_g:")
sp.pprint(sp.Eq(lhs_mass_ug, rhs_mass_ug))

# 2. Уравнение импульса (Момент 1 от Eq_eps1)
print("\n--- Анализ порядка epsilon^1 (Momentum) ---")
LHS_mom_x_eps1 = 0
RHS_mom_x_eps1 = Fx * dt_sym / (2 * tau)
LHS_mom_y_eps1 = 0
RHS_mom_y_eps1 = Fy * dt_sym / (2 * tau)

for i in range(Q):
    term_l = LHS_eps1.subs({c_x_sym: c_x[i], c_y_sym: c_y[i], f_0: f_eq[i]})
    LHS_mom_x_eps1 += term_l * c_x[i]
    LHS_mom_y_eps1 += term_l * c_y[i]

    term_r = RHS_eps1.subs({f_1: 0, S_1: Guo_S_list[i]})
    # f_1 зануляем в цикле, но его вклад (sum c f1 = -m F dt) уже учтен в начальном значении RHS_mom_x_eps1 = Fx*dt/(2tau)
    # sum(c * (-1/tau * f1)) = -1/tau * (-F*dt/2) = F*dt/(2tau)
    RHS_mom_x_eps1 += term_r * c_x[i]
    RHS_mom_y_eps1 += term_r * c_y[i]

print(f"Импульс X:")
lhs_mom_x_simple = sp.simplify(LHS_mom_x_eps1 / dt_sym)
rhs_mom_x_simple = sp.simplify(RHS_mom_x_eps1 / dt_sym)
sp.pprint(sp.Eq(lhs_mom_x_simple, rhs_mom_x_simple))
print(f"Импульс Y:")
lhs_mom_y_simple = sp.simplify(LHS_mom_y_eps1 / dt_sym)
rhs_mom_y_simple = sp.simplify(RHS_mom_y_eps1 / dt_sym)
sp.pprint(sp.Eq(lhs_mom_y_simple, rhs_mom_y_simple))

# Перезаписываем уравнения импульса в терминах u_g
lhs_mom_x_ug = sp.simplify(lhs_mom_x_simple.subs(subs_to_ug))
rhs_mom_x_ug = sp.simplify(rhs_mom_x_simple.subs(subs_to_ug))

lhs_mom_y_ug = sp.simplify(lhs_mom_y_simple.subs(subs_to_ug))
rhs_mom_y_ug = sp.simplify(rhs_mom_y_simple.subs(subs_to_ug))

print("\nИмпульс X в терминах u_g:")
sp.pprint(sp.Eq(lhs_mom_x_ug, rhs_mom_x_ug))

print("\nИмпульс Y в терминах u_g:")
sp.pprint(sp.Eq(lhs_mom_y_ug, rhs_mom_y_ug))

print("Подсчет подстановок.")

# 1. Вычисляем Тензор Потока Импульса нулевого порядка (Pi_0)
# Pi_alpha_beta = sum(c_alpha * c_beta * f_eq)
print("Вычисление тензора потока импульса (Pi^0)...")
Pi_xx_0 = sp.simplify(sum(c_x[i] * c_x[i] * f_eq[i] for i in range(Q)))
Pi_xy_0 = sp.simplify(sum(c_x[i] * c_y[i] * f_eq[i] for i in range(Q)))
Pi_yy_0 = sp.simplify(sum(c_y[i] * c_y[i] * f_eq[i] for i in range(Q)))
print("\nPi^0_XX :")
sp.pprint(Pi_xx_0)
print("\nPi^0_XY :")
sp.pprint(Pi_xy_0)
print("\nPi^0_YY :")
sp.pprint(Pi_yy_0)


# 2. Формируем подстановки из уравнений Эйлера (из порядка eps^1)
# Нам нужно выразить dt1(rho) и dt1(u), dt1(v) через пространственные производные.

div_rho_u = dx1 * (rho * u) + dy1 * (rho * v)
dt1_rho_subs = -div_rho_u - dx1 * (Fx * dt_sym / 2) - dy1 * (Fy * dt_sym / 2)

# Уравнение импульса Эйлера: dt1(rho*u) + div(Pi) = F - dt1(deltaT*F / 2)
# Для упрощения вывода берем классическую форму Эйлера:
# dt1(rho*u) = - (dx1*Pi_xx + dy1*Pi_xy) + Fx
# dt1(rho*v) = - (dx1*Pi_xy + dy1*Pi_yy) + Fy

dt1_rhou_subs = - (dx1 * Pi_xx_0 + dy1 * Pi_xy_0) + Fx - dt1 * (Fx * dt_sym / 2)
dt1_rhov_subs = - (dx1 * Pi_xy_0 + dy1 * Pi_yy_0) + Fy - dt1 * (Fy * dt_sym / 2)

# Выражаем чистые производные скорости dt1(u) и dt1(v)
# dt1(rho*u) = rho*dt1(u) + u*dt1(rho)  =>  dt1(u) = (dt1(rho*u) - u*dt1(rho)) / rho
dt1_u_subs = sp.simplify((dt1_rhou_subs - u * dt1_rho_subs) / rho)
dt1_v_subs = sp.simplify((dt1_rhov_subs - v * dt1_rho_subs) / rho)

print("Подстановки Эйлера сформированы.")

print("\n--- Анализ порядка epsilon^2 (Viscous / Navier-Stokes) ---")


def time_derivative_euler(expr):
    """
    Вычисляет полную производную d/dt1 от выражения expr(rho, u, v),
    используя уравнения Эйлера для замены dt1(rho), dt1(u), dt1(v).
    """
    # Полный дифференциал
    diff_rho = sp.diff(expr, rho) * dt1_rho_subs
    diff_u = sp.diff(expr, u) * dt1_u_subs
    diff_v = sp.diff(expr, v) * dt1_v_subs

    return sp.simplify(diff_rho + diff_u + diff_v)


def evaluate_dt1_operators(expr):
    """
    Интерпретирует символы dt1 как оператор дифференцирования.
    Если в слагаемом есть dt1^k, то применяется k раз производная time_derivative_euler
    к остальной части слагаемого.
    """
    # Раскрываем скобки, чтобы получить сумму одночленов
    expanded = sp.expand(expr)

    if expanded.func == sp.Add:
        args = expanded.args
    else:
        args = [expanded]

    result = 0
    for term in args:
        if term.has(dt1):
            # Определяем степень dt1
            # term = coeff * dt1**k
            # as_coeff_exponent возвращает (coeff, exponent) для заданного символа
            coeff, exponent = term.as_coeff_exponent(dt1)

            # Применяем производную exponent раз
            deriv = coeff
            for _ in range(exponent):
                deriv = time_derivative_euler(deriv)

            result += deriv
        else:
            result += term

    return sp.simplify(result)


print("Формируем полное выражение для LHS второго порядка...")

# 1. Подставляем f_1 в выражение LHS_eps2
# f_1 = -tau * dt * (D1 f_0 - S_1)
# LHS_eps2 содержит члены с f_1 и f_0
f_1_expr = -tau * dt_sym * (D1 * f_0 - S_1)
LHS_eps2_full = LHS_eps2.subs(f_1, f_1_expr)

print("Суммируем уравнение непрерывности (Момент 0) с учетом подстановок...")
LHS_Mass_Accumulator = 0

for i in range(Q):
    # Подставляем конкретные c_i, f_eq, S_i для каждого направления
    term = LHS_eps2_full.subs({
        c_x_sym: c_x[i],
        c_y_sym: c_y[i],
        f_0: f_eq[i],
        S_1: Guo_S_list[i]
    })
    LHS_Mass_Accumulator += term

# Это самый тяжелый этап - раскрытие всех dt1
LHS_Mass_Calculated = evaluate_dt1_operators(LHS_Mass_Accumulator)

print("Уравнение непрерывности (после упрощения):")
# Делим на dt, так как уравнение имеет вид dt * (... ) = 0
Continuity_Eq_Order2 = sp.simplify(LHS_Mass_Calculated / dt_sym)
sp.pprint(sp.Eq(Continuity_Eq_Order2, 0))


print("\n--- Уравнение непрерывности второго порядка в терминах скорости по Гуо (u_g) ---")
# Переписываем Pi^0 в терминах u_g (обратная подстановка Guo-скорости)
Pi_xx_0_ug = sp.simplify(Pi_xx_0.subs(subs_to_ug))
Pi_xy_0_ug = sp.simplify(Pi_xy_0.subs(subs_to_ug))
Pi_yy_0_ug = sp.simplify(Pi_yy_0.subs(subs_to_ug))

print("\nPi^0_XX в терминах u_g:")
sp.pprint(Pi_xx_0_ug)
print("\nPi^0_XY в терминах u_g:")
sp.pprint(Pi_xy_0_ug)
print("\nPi^0_YY в терминах u_g:")
sp.pprint(Pi_yy_0_ug)

# 1. Подстановки Эйлера в терминах u_g
# dt1(rho) + div (rho u_g) = 0  =>  dt1(rho) = -div (rho u_g)
div_rho_ug = dx1 * (rho * u_g_sym) + dy1 * (rho * v_g_sym)
dt1_rho_subs_guo = -div_rho_ug

# Импульс Эйлера: dt1(rho*u_g) + div(Pi^0(u_g)) = F
dt1_rhou_subs_guo = - (dx1 * Pi_xx_0_ug + dy1 * Pi_xy_0_ug) + Fx
dt1_rhov_subs_guo = - (dx1 * Pi_xy_0_ug + dy1 * Pi_yy_0_ug) + Fy

# dt1(rho*u_g) = rho*dt1(u_g) + u_g*dt1(rho)
dt1_u_subs_guo = sp.simplify((dt1_rhou_subs_guo - u_g_sym * dt1_rho_subs_guo) / rho)
dt1_v_subs_guo = sp.simplify((dt1_rhov_subs_guo - v_g_sym * dt1_rho_subs_guo) / rho)


def time_derivative_euler_guo(expr):
    """d/dt1 от expr(rho, u_g, v_g) с использованием Эйлера в терминах u_g."""
    diff_rho = sp.diff(expr, rho) * dt1_rho_subs_guo
    diff_u = sp.diff(expr, u_g_sym) * dt1_u_subs_guo
    diff_v = sp.diff(expr, v_g_sym) * dt1_v_subs_guo
    return sp.simplify(diff_rho + diff_u + diff_v)


def evaluate_dt1_operators_guo(expr):
    """Аналог evaluate_dt1_operators, но через time_derivative_euler_guo."""
    expanded = sp.expand(expr)
    if expanded.func == sp.Add:
        args = expanded.args
    else:
        args = [expanded]

    result = 0
    for term in args:
        if term.has(dt1):
            coeff, exponent = term.as_coeff_exponent(dt1)
            deriv = coeff
            for _ in range(exponent):
                deriv = time_derivative_euler_guo(deriv)
            result += deriv
        else:
            result += term

    return sp.simplify(result)


# 2. Строим новую сумму нулевого момента уже в терминах u_g
LHS_Mass_Accumulator_guo = 0
for i in range(Q):
    term_guo = LHS_eps2_full.subs({
        c_x_sym: c_x[i],
        c_y_sym: c_y[i],
        f_0: f_eq[i],
        S_1: Guo_S_list[i]
    })
    # Обратная подстановка: выражаем u, v через u_g
    term_guo = term_guo.subs(subs_to_ug)
    LHS_Mass_Accumulator_guo += term_guo

LHS_Mass_Calculated_guo = evaluate_dt1_operators_guo(LHS_Mass_Accumulator_guo)
Continuity_Eq_Order2_guo = sp.simplify(LHS_Mass_Calculated_guo / dt_sym)
print("Уравнение непрерывности (Гуо переменные) (после упрощения):")
sp.pprint(sp.Eq(Continuity_Eq_Order2_guo, 0))


# --- УРАВНЕНИЕ ИМПУЛЬСА ВТОРОГО ПОРЯДКА (eps^2) ---

print("\n--- Уравнение импульса второго порядка в терминах скорости по Гуо (u_g) ---")
print("Используем метод прямого вычисления моментов для избежания ошибок с оператором D1^2.")

# dt2(rho*u_g) = div(Sigma_ab)

# 1. Вычисляем моменты высшего порядка для f_eq (Q = c*c*c*f_eq)
# Нам нужны Q_xxx, Q_xxy, Q_xyy, Q_yyy
# Используем subs_to_ug, чтобы получить выражения через u_g

Q_xxx_0 = sp.simplify(sum(c_x[i] * c_x[i] * c_x[i] * f_eq[i] for i in range(Q)))
Q_xxy_0 = sp.simplify(sum(c_x[i] * c_x[i] * c_y[i] * f_eq[i] for i in range(Q)))
Q_xyy_0 = sp.simplify(sum(c_x[i] * c_y[i] * c_y[i] * f_eq[i] for i in range(Q)))
Q_yyy_0 = sp.simplify(sum(c_y[i] * c_y[i] * c_y[i] * f_eq[i] for i in range(Q)))

Q_xxx_0_ug = sp.simplify(Q_xxx_0.subs(subs_to_ug))
Q_xxy_0_ug = sp.simplify(Q_xxy_0.subs(subs_to_ug))
Q_xyy_0_ug = sp.simplify(Q_xyy_0.subs(subs_to_ug))
Q_yyy_0_ug = sp.simplify(Q_yyy_0.subs(subs_to_ug))
print("\nQ^0_XXX :")
sp.pprint(Q_xxx_0_ug)
print("\nQ^0_XXY :")
sp.pprint(Q_xxy_0_ug)
print("\nQ^0_XYY :")
sp.pprint(Q_xyy_0_ug)
print("\nQ^0_YYY :")
sp.pprint(Q_yyy_0_ug)



# 2. Вычисляем моменты от источника силы (Lambda = c*c*S)
Lambda_xx = sp.simplify(sum(c_x[i] * c_x[i] * Guo_S_list[i] for i in range(Q)))
Lambda_xy = sp.simplify(sum(c_x[i] * c_y[i] * Guo_S_list[i] for i in range(Q)))
Lambda_yy = sp.simplify(sum(c_y[i] * c_y[i] * Guo_S_list[i] for i in range(Q)))

# Lambda уже зависит от F и u (которые станут u_g после подстановки subs_to_ug)
Lambda_xx_ug = Lambda_xx.subs(subs_to_ug)
Lambda_xy_ug = Lambda_xy.subs(subs_to_ug)
Lambda_yy_ug = Lambda_yy.subs(subs_to_ug)

# 3. Вычисляем компоненты тензора напряжений первого порядка Pi^(1)
# Вместо прямого дифференцирования моментов (которое дает паразитные кубические члены u^3),
# используем аналитическое выражение для Pi^(1) в приближении малых чисел Маха (low-Mach approximation).
# Pi^(1)_ab = -tau * dt * ( c_s^2 * rho * (d_a u_b + d_b u_a) + (u_a F_b + u_b F_a) - Lambda_ab )
# Это выражение получается из (dt1 Pi^0 + div Q^0), если отбросить O(u^3) и использовать Pi^0 = rho cs2 I + rho uu.

print("Используем аналитическую формулу для Pi^(1) (low-Mach) для устранения нефизичных членов...")

term_visc_xx = rho * cs2 * (2 * dx1 * u_g_sym)
term_visc_xy = rho * cs2 * (dy1 * u_g_sym + dx1 * v_g_sym)
term_visc_yy = rho * cs2 * (2 * dy1 * v_g_sym)

term_force_xx = 2 * u_g_sym * Fx
term_force_xy = u_g_sym * Fy + v_g_sym * Fx
term_force_yy = 2 * v_g_sym * Fy

Pi1_xx = -tau * dt_sym * (term_visc_xx + term_force_xx - Lambda_xx_ug)
Pi1_xy = -tau * dt_sym * (term_visc_xy + term_force_xy - Lambda_xy_ug)
Pi1_yy = -tau * dt_sym * (term_visc_yy + term_force_yy - Lambda_yy_ug)

# 4. Формируем тензор напряжений Sigma для уравнения Навье-Стокса
# sigma = -(1 - 1/(2tau)) * Pi1 - (dt/4)(C + C^T)
# Поскольку C (Lambda) симметричен, то (dt/4)(2C) = (dt/2)C
# Lambda (наш C) = (1 - 1/2tau)(uF + Fu)

factor_sigma = -(1 - 1 / (2 * tau))

# Добавляем поправку - (dt/2) * Lambda
Sigma_xx = factor_sigma * Pi1_xx - (dt_sym / 2) * Lambda_xx_ug
Sigma_xy = factor_sigma * Pi1_xy - (dt_sym / 2) * Lambda_xy_ug
Sigma_yy = factor_sigma * Pi1_yy - (dt_sym / 2) * Lambda_yy_ug

def convert_symbols_to_derivatives(expr):
    """
    Преобразует символические множители dx1, dy1 в объекты sp.Derivative для красивого вывода.
    Предполагается, что dx1, dy1 действуют на весь остальной коэффициент одночлена.
    """
    expr_expanded = sp.expand(expr)
    if expr_expanded == 0:
        return 0

    args = expr_expanded.args if expr_expanded.func == sp.Add else [expr_expanded]
    new_terms = []

    for term in args:
        # Извлекаем степени dx1 и dy1
        # as_coeff_exponent работает для конкретного символа
        
        # Получаем множитель (coeff_dx) и степень (pow_x) для dx1
        coeff_dx, pow_x = term.as_coeff_exponent(dx1)
        # В coeff_dx всё ещё может быть dy1.
        
        # Теперь из coeff_dx извлекаем dy1
        final_coeff, pow_y = coeff_dx.as_coeff_exponent(dy1)
        
        # final_coeff - это выражение, на которое действуют производные
        if pow_x == 0 and pow_y == 0:
            new_terms.append(final_coeff)
        else:
            # Формируем производную
            # Порядок переменных: сначала x, потом y (можно наоборот, неважно)
            deriv_args = []
            if pow_x > 0:
                deriv_args.extend([x] * pow_x)
            if pow_y > 0:
                deriv_args.extend([y] * pow_y)
            
            # Создаем объект производной
            # simplify=False, чтобы не раскрывать производные (если бы final_coeff было выражением, которое sympy умеет дифференцировать)
            # Но здесь final_coeff содержит u(x,y,t) как абстрактную функцию, так что sympy просто запишет Derivative.
            new_term = sp.Derivative(final_coeff, *deriv_args)
            new_terms.append(new_term)
            
    return sum(new_terms)

RHS_MomX_Order2 = sp.simplify(dx1 * Sigma_xx + dy1 * Sigma_xy)
RHS_MomY_Order2 = sp.simplify(dx1 * Sigma_xy + dy1 * Sigma_yy)

print("\n--- АНАЛИЗ ВЯЗКИХ ЧЛЕНОВ ---")
# Определяем кинематическую вязкость nu
# nu = cs^2 * (tau - 1/2) * dt = 1/3 * (tau - 1/2) * dt
nu = sp.Symbol('nu')
# Выражение для nu через tau, dt
nu_expr = cs2 * (tau - sp.Rational(1, 2)) * dt_sym

# Заменим (tau - 1/2)dt/3 на nu
# Для этого выразим tau через nu: 
# nu = (tau - 1/2)dt/3 => 3nu/dt = tau - 1/2 => tau = 3nu/dt + 1/2
tau_subs = 3 * nu / dt_sym + sp.Rational(1, 2)

RHS_MomX_Nu = sp.simplify(RHS_MomX_Order2.subs(tau, tau_subs))
RHS_MomY_Nu = sp.simplify(RHS_MomY_Order2.subs(tau, tau_subs))

print("Импульс X (через nu):")
sp.pprint(sp.Eq(sp.Symbol('∂(ρ u)/∂t₂'), convert_symbols_to_derivatives(RHS_MomX_Nu)))

print("Импульс Y (через nu):")
sp.pprint(sp.Eq(sp.Symbol('∂(ρ v)/∂t₂'), convert_symbols_to_derivatives(RHS_MomY_Nu)))

def convert_symbols_to_derivatives(expr):
    """
    Преобразует символические множители dx1, dy1 в объекты sp.Derivative для красивого вывода.
    Предполагается, что dx1, dy1 действуют на весь остальной коэффициент одночлена.
    """
    expr_expanded = sp.expand(expr)
    if expr_expanded == 0:
        return 0

    args = expr_expanded.args if expr_expanded.func == sp.Add else [expr_expanded]
    new_terms = []

    for term in args:
        # Извлекаем степени dx1 и dy1
        # as_coeff_exponent работает для конкретного символа
        
        # Получаем множитель (coeff_dx) и степень (pow_x) для dx1
        coeff_dx, pow_x = term.as_coeff_exponent(dx1)
        # В coeff_dx всё ещё может быть dy1.
        
        # Теперь из coeff_dx извлекаем dy1
        final_coeff, pow_y = coeff_dx.as_coeff_exponent(dy1)
        
        # final_coeff - это выражение, на которое действуют производные
        if pow_x == 0 and pow_y == 0:
            new_terms.append(final_coeff)
        else:
            # Формируем производную
            # Порядок переменных: сначала x, потом y (можно наоборот, неважно)
            deriv_args = []
            if pow_x > 0:
                deriv_args.extend([x] * pow_x)
            if pow_y > 0:
                deriv_args.extend([y] * pow_y)
            
            # Создаем объект производной
            # simplify=False, чтобы не раскрывать производные (если бы final_coeff было выражением, которое sympy умеет дифференцировать)
            # Но здесь final_coeff содержит u(x,y,t) как абстрактную функцию, так что sympy просто запишет Derivative.
            new_term = sp.Derivative(final_coeff, *deriv_args)
            new_terms.append(new_term)
            
    return sum(new_terms)


print("\n--- ШАГ 5: ВОССТАНОВЛЕНИЕ ПОЛНЫХ УРАВНЕНИЙ НАВЬЕ-СТОКСА ---")

# 1. Уравнение непрерывности
# d/dt rho + div(rho u) = 0
# Для вывода используем то, что уравнение непрерывности не получило поправок на втором шаге (0 = 0)
# (после корректного сокращения).
# Поэтому выводим стандартную форму.

print("\n--- Итоговое уравнение непрерывности ---")
# LHS: ∂ρ/∂t
# RHS: -∇·(ρu)
LHS_Cont_Final = sp.Derivative(rho, t)
RHS_Cont_Final = -(sp.Derivative(rho * u_g_sym, x) + sp.Derivative(rho * v_g_sym, y))
sp.pprint(sp.Eq(LHS_Cont_Final, RHS_Cont_Final))


# 2. Уравнение импульса
# Структура:
# ∂(ρu)/∂t + ∇·(ρuu) = -∇p + F + ∇·σ
# Переносим конвективный член направо:
# ∂(ρu)/∂t = -∇·(ρuu) - ∇p + F + ∇·σ

print("\n--- Итоговое уравнение импульса X (Навье-Стокса) ---")

# Строим уравнение из КОМПОНЕНТ, вычисленных программой ранее
# Уровень eps^1 (Euler): lhs_mom_x_ug = rhs_mom_x_ug
# lhs_mom_x_ug содержит слагаемое с dt1 (временная производная) и пространственные члены.
# Нам нужно выразить dt1(...) = RHS - Spatial
# Pi^0 = rho uu + p I уже внутри lhs_mom_x_ug

# 1. Выделяем пространственную часть Эйлера (полагаем d/dt1 = 0)
Euler_Spatial_Part_X = lhs_mom_x_ug.subs(dt1, 0)

# 2. Выделяем правую часть Эйлера (Сила Fx)
Euler_RHS_X = rhs_mom_x_ug

# 3. Вязкая часть (уже вычислена как RHS_MomX_Nu)
Viscous_Part_X = RHS_MomX_Nu

# 4. Собираем полную правую часть:
# d/dt(rho u) = [Euler_RHS - Euler_Spatial] + Viscous_Part
Total_RHS_X_Computed = Euler_RHS_X - Euler_Spatial_Part_X + Viscous_Part_X

# Преобразуем символы dx1, dy1 в производные для красивого вывода
Total_RHS_X_Pretty = convert_symbols_to_derivatives(Total_RHS_X_Computed)
LHS_Mom_X = sp.Derivative(rho * u_g_sym, t)

sp.pprint(sp.Eq(LHS_Mom_X, Total_RHS_X_Pretty))


print("\n--- Итоговое уравнение импульса Y (Навье-Стокса) ---")

Euler_Spatial_Part_Y = lhs_mom_y_ug.subs(dt1, 0)
Euler_RHS_Y = rhs_mom_y_ug
Viscous_Part_Y = RHS_MomY_Nu

Total_RHS_Y_Computed = Euler_RHS_Y - Euler_Spatial_Part_Y + Viscous_Part_Y

Total_RHS_Y_Pretty = convert_symbols_to_derivatives(Total_RHS_Y_Computed)
LHS_Mom_Y = sp.Derivative(rho * v_g_sym, t)

sp.pprint(sp.Eq(LHS_Mom_Y, Total_RHS_Y_Pretty))


print("\nКоэффициенты:")
print(f"p = ρ c_s^2 = ρ/3")
print(f"ν = c_s^2 (τ - 1/2) Δt")
print(f"u_g = физическая скорость жидкости")



