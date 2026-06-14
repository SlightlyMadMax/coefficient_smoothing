# Решатель двумерных задач плавления и затвердевания с естественной конвекцией

[English](README.md) | **Русский**

Численный решатель двумерных задач плавления/затвердевания с естественной конвекцией в
расплаве. Сочетает формулировку «функция тока–вихрь скорости» для несжимаемых уравнений
Навье–Стокса с методом эффективной теплоёмкости для расчёта теплопереноса с фазовым
переходом.

Код полностью написан на Python и имеет модульную архитектуру: физическая модель, схемы
дискретизации, линейные решатели и оркестрация расчёта разделены и взаимозаменяемы.
Критичные по производительности участки ускорены JIT-компиляцией
[Numba](https://numba.pydata.org/), остальные операции над массивами векторизованы
средствами NumPy. Разреженные эллиптические задачи решаются с помощью SciPy / PyAMG
(с опциональным GPU-бэкендом на CuPy).

## Физическая модель

Поле скорости восстанавливается из функции тока:
$v_x = \partial\psi/\partial y$, $v_y = -\partial\psi/\partial x$.

Все уравнения записаны в безразмерном виде с использованием характерной длины $L$,
конвективной скорости $V = \sqrt{g \beta \Delta T L}$ и безразмерной температуры
$\theta = (T - T_\text{ref})/\Delta T$.

- **Перенос вихря** (со штрафным членом $S$ метода фиктивных областей для твёрдой фазы):

$$\frac{\partial \omega}{\partial t} + \frac{\partial}{\partial x}\left(\omega \frac{\partial \psi}{\partial y}\right) - \frac{\partial}{\partial y}\left(\omega \frac{\partial \psi}{\partial x}\right) = \frac{1}{Re}\nabla^2\omega + \frac{Gr}{Re^2}\frac{\partial\theta}{\partial x} - \nabla \cdot (S \nabla\psi)$$

- **Связь функции тока и вихря**:

$$\nabla^2\psi = -\omega$$

- **Уравнение энергии** (метод эффективной теплоёмкости):

$$c_\text{eff}(\theta)\left(\frac{\partial\theta}{\partial t} + \frac{\partial(v_x \theta)}{\partial x} + \frac{\partial(v_y \theta)}{\partial y}\right) = \frac{1}{Pe}\nabla\cdot\left(k_\text{eff}(\theta)\nabla\theta\right)$$

где эффективная теплоёмкость поглощает выделение скрытой теплоты:

$$c_\text{eff}(\theta) = \frac{\tilde{c}(\theta)}{c_\text{ref}} + \frac{\delta_\Delta(\theta - \theta_m)}{Ste}$$

$\delta_\Delta$ — гладкая аппроксимация дельта-функции Дирака, размазывающая границу
раздела твёрдой и жидкой фаз по двухфазной зоне полуширины $\Delta$ (в безразмерных
единицах температуры). Штраф $S$ — гладкая функция $\theta$, обнуляющая скорость в
твёрдой фазе.

## Численный метод

Все уравнения дискретизируются конечными разностями на равномерной прямоугольной сетке.
На каждом шаге по времени сначала обновляется температура, затем течение:

1. **Теплоперенос** — обновление $\theta$, а вместе с ним фазового поля, штрафного и подъёмного коэффициентов.
2. **Течение жидкости** — обновление $\omega$ и $\psi$ по новой температуре.

### Теплоперенос

Фазовый переход учитывается методом эффективной теплоёмкости: скрытая теплота включается
в $c_\text{eff}$ через сглаженную дельта-функцию по двухфазной зоне (см.
[Физическую модель](#физическая-модель)). Формы сглаживания для ступенчатой и
дельта-функций взаимозаменяемы (`StepScheme`, `DeltaScheme`), как и правило вычисления
теплопроводностей $k_\text{eff}$ на гранях — среднее арифметическое, среднее
гармоническое или вычисление по температуре на грани (`KFaceMethod`). Конвективный член
дискретизируется выбираемой схемой (`ConvectiveTermForm`) — от центральных разностей и
противопоточной аппроксимации до схемы с отложенной коррекцией (TVD). Уравнение энергии
интегрируется методом переменных направлений (ADI) — схемами Писмена–Рэкфорда,
Дугласа–Рэкфорда или локально-одномерной; также доступны полностью неявный и явный
решатели.

### Течение жидкости

Твёрдая фаза учитывается методом фиктивных областей (штрафным методом): член $S$
становится большим в твёрдой фазе, обнуляя там скорость; его форма выбираема
(`PenaltyTermForm`). Доступны два решателя уравнений Навье–Стокса:

- **`BCCorrectionNVSolver`** (используется по умолчанию во всех примерах) исключает
  внутренние итерации между уравнениями на вихрь и функцию тока. Он реализует схему с
  поправкой по граничному условию (Самарский, Вабищевич): граничное условие на вихрь
  вкладывается в уравнение четвёртого порядка для $\psi$, которое продвигается по времени
  расщеплением Дугласа–Рэкфорда («предиктор-корректор»). Это позволяет использовать
  существенно больший устойчивый шаг по времени, что важно при больших числах Рэлея,
  характерных для таяния льда. На шаге-корректоре решается *модифицированное*
  эллиптическое уравнение, а не обычное уравнение Пуассона, поэтому требуется решатель
  функции тока `AMG`, `CG` или `CG_GPU`.

- **`IterativeNavierStokesSolver`** — классическая схема: решить уравнение переноса вихря,
  решить $\nabla^2\psi = -\omega$, повторять до сходимости. Решатели функции тока `SOR` и
  `MATRIX_SWEEP` работают только с ней.

## Установка

Требуется Python 3.11 или 3.12.

```bash
pip install -r requirements.txt
# или, через Poetry:
poetry install
```

> **Поддержка GPU** требует CuPy с рантаймом CUDA 12.x.

## Примеры

Готовые к запуску расчёты находятся в `src/examples/`, каждый в своём подкаталоге с
файлами `config.json` и `run.py` (точка входа):

| Каталог | Задача |
|---|---|
| `stefan/` | Чисто кондуктивная задача Стефана (без течения) |
| `gallium/` | Плавление галлия с естественной конвекцией |
| `octadecane/` | Плавление n-октадекана в дифференциально нагреваемой полости |
| `water_convection/` | Естественная конвекция в жидкой воде |
| `water_freezing/` | Замерзание воды с конвекцией |
| `horizontal_layer/` | Плавление горизонтального слоя |
| `icicle/` | Рост сосульки |
| `crevasse/` | Таяние трещины (ледниковой) |
| `air/` | Эталонный расчёт конвекции воздуха |

Каждый `run.py` демонстрирует полную настройку решателя для соответствующего материала и
геометрии и может служить шаблоном для новых задач.

## Конфигурация

Расчёты настраиваются через JSON-файлы, загружаемые в `ExperimentConfig`:

```python
from src.parameters.config import ExperimentConfig

cfg = ExperimentConfig.load_from_file("parameter_sets/my_case/config.json")
```

Файл конфигурации задаёт:

| Поле | Описание |
|---|---|
| `geometry` | `width`, `height`, `end_time`, `n_x`, `n_y`, `n_t` |
| `u_ref` | Опорная температура [K] |
| `delta_u` | Характерный перепад температур [K] |
| `l` | Характерная длина [м] |
| `delta` | Полуширина двухфазной зоны для теплопереноса [K] (необязательно; оценивается автоматически, если не задано) |
| `delta_flow` | Полуширина двухфазной зоны для течения [K] (необязательно) |
| `epsilon` | Параметр штрафа метода фиктивных областей |
| `material_props` | См. `MaterialProperties` ниже |

Поля `MaterialProperties`: `u_pt`, `specific_heat_liquid`, `specific_heat_solid`,
`specific_latent_heat`, `density_liquid`, `density_solid`,
`thermal_conductivity_liquid`, `thermal_conductivity_solid`,
`dynamic_viscosity`, `volumetric_thermal_exp`, `density_poly_coeffs` (необязательно).

`ExperimentConfig` вычисляет безразмерные числа (`Re`, `Gr`, `Ra`, `Pr`, `Pe`, `Ste`) по
свойствам материала и характерным масштабам, а также предоставляет масштабированные шаги
сетки.

## Использование

### Запуск расчёта

```python
from src.core.boundary_conditions import BoundaryConditions
from src.core.runner import SimulationState, ExperimentRunner
from src.fluid_dynamics.init_values import initialize_stream_function, initialize_vorticity, initialize_velocity
from src.fluid_dynamics.solvers import VorticitySolverName, StreamFunctionSolverName
from src.fluid_dynamics.solvers.bc_correction_solver_factory import BCCorrectionNVSolver
from src.fluid_dynamics.solvers.vorticity_solvers.base_solver import PenaltyTermForm
from src.heat_transfer.init_values import init_temperature, DomainShape
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.heat_transfer.solvers.heat_transfer_solvers.base_solver import KFaceMethod
from src.heat_transfer.coefficient_smoothing.coefficients import StepScheme, DeltaScheme
from src.convective_operators import ConvectiveTermForm
from src.parameters.config import ExperimentConfig
from src.utils.boundary_conditions import const_dirichlet_condition, const_neumann_condition

cfg = ExperimentConfig.load_from_file("parameter_sets/my_case/config.json")
geometry = cfg.geometry

u_bcs = BoundaryConditions(
    left=const_dirichlet_condition(geometry.n_y, value=1.0),
    right=const_dirichlet_condition(geometry.n_y, value=0.0),
    top=const_neumann_condition(geometry.n_x, value=0.0),
    bottom=const_neumann_condition(geometry.n_x, value=0.0),
)
sf_bcs = BoundaryConditions(
    left=const_dirichlet_condition(geometry.n_y, value=0.0),
    right=const_dirichlet_condition(geometry.n_y, value=0.0),
    top=const_dirichlet_condition(geometry.n_x, value=0.0),
    bottom=const_dirichlet_condition(geometry.n_x, value=0.0),
)

u  = init_temperature(cfg=cfg, bcs=u_bcs, shape=DomainShape.UNIFORM_SOLID, solid_temp=cfg.material_props.u_pt - 1.0)
sf = initialize_stream_function(geometry=geometry, bcs=sf_bcs)
w  = initialize_vorticity(geometry=geometry)
v_x, v_y = initialize_velocity(geometry=geometry)

heat_solver = HeatTransferSolver(
    cfg=cfg,
    bcs=u_bcs,
    solver_name=HeatTransferSolverName.PEACEMAN_RACHFORD,
    convective_term_form=ConvectiveTermForm.DEFERRED_CORRECTION,
    step_scheme=StepScheme.ERF,
    delta_scheme=DeltaScheme.GAUSS,
    k_face_method=KFaceMethod.FROM_TEMP,
    max_iters=1,
    tolerance=1e-6,
    urf=1.0,
)

navier_solver = BCCorrectionNVSolver(
    cfg=cfg,
    sf_bcs=sf_bcs,
    vorticity_solver_name=VorticitySolverName.PEACEMAN_RACHFORD,
    stream_function_solver_name=StreamFunctionSolverName.AMG,
    convective_term_form=ConvectiveTermForm.DIVERGENT_CENTRAL,
    penalty_term_form=PenaltyTermForm.LINEAR,
    vorticity_bc_order=2,
)

state = SimulationState(u=u, sf=sf, w=w, v_x=v_x, v_y=v_y)

runner = ExperimentRunner(
    cfg=cfg,
    state=state,
    heat_solver=heat_solver,
    navier_solver=navier_solver,
    checkpoints_dir="data/my_case",
    save_at={500, 1000, 2000},
)
runner.run()
```

### Возобновление из контрольной точки

```python
runner = ExperimentRunner.from_checkpoint(
    checkpoint_path="data/my_case/checkpoint_1000.npz",
    cfg=cfg,
    heat_solver=heat_solver,
    navier_solver=navier_solver,
    checkpoints_dir="data/my_case",
    save_at={2000},
)
runner.run()
```

## Параметры решателей

### Решатели для вихря (`VorticitySolverName`)

| Имя | Схема | Примечания |
|---|---|---|
| `PEACEMAN_RACHFORD` | ADI Писмена–Рэкфорда | Рекомендуется |
| `DOUGLAS_RACHFORD` | ADI Дугласа–Рэкфорда | Безусловно устойчива |
| `LOC_ONE_DIM` | Локально-одномерная | Полный шаг по каждому направлению |
| `EXPLICIT` | Явная (Эйлера) | Устойчива только при малом шаге по времени |

### Решатели для функции тока (`StreamFunctionSolverName`)

| Имя | Метод | Примечания |
|---|---|---|
| `AMG` | Алгебраический многосеточный (PyAMG) | Самый быстрый на больших сетках; рекомендуется |
| `CG` | Метод сопряжённых градиентов (SciPy) | Хорошая альтернатива |
| `CG_GPU` | Метод сопряжённых градиентов (CuPy) | Требует CUDA 12.x |
| `SOR` | Последовательная верхняя релаксация | Только для классического уравнения Пуассона $\nabla^2\psi = -\omega$ |
| `MATRIX_SWEEP` | Прямая трёхдиагональная прогонка | Только для классического уравнения Пуассона $\nabla^2\psi = -\omega$ |

### Решатели для теплопереноса (`HeatTransferSolverName`)

| Имя | Схема |
|---|---|
| `PEACEMAN_RACHFORD` | ADI Писмена–Рэкфорда |
| `DOUGLAS_RACHFORD` | ADI Дугласа–Рэкфорда |
| `LOC_ONE_DIM` | Локально-одномерная |
| `FULLY_IMPLICIT` | Полностью неявная |
| `EXPLICIT` | Явная (Эйлера) |

### Формы конвективного члена (`ConvectiveTermForm`)

| Имя | Описание |
|---|---|
| `DIVERGENT_CENTRAL` | Центральная разность от $\partial(v_i\phi)/\partial x_i$ |
| `NON_DIVERGENT_CENTRAL` | Центральная разность от $v_i\,\partial\phi/\partial x_i$ |
| `SYMMETRIC` | Среднее дивергентной и недивергентной форм |
| `UPWIND_NC` | Противопоточная 1-го порядка в узлах |
| `UPWIND_FC` | Противопоточная 1-го порядка на гранях |
| `DEFERRED_CORRECTION` | Противопоточная основа + поправка высокого порядка (только теплоперенос) |

### Схемы ступенчатой и дельта-функций

**Ступенчатая** (`StepScheme`): `ERF`, `HYPER`, `LINEAR`, `CONST`, `JUMP`

**Дельта** (`DeltaScheme`): `GAUSS`, `HYPER`, `PARABOLIC`, `BOX`

### Формы штрафного члена (`PenaltyTermForm`)

| Имя | Формула |
|---|---|
| `JUMP` | $C \cdot \mathbf{1}_{u \le u_0}$ |
| `LINEAR` | $C \cdot \frac{1}{2}(1 - \tanh(\tilde{u}/\Delta))$ |
| `QUADRATIC` | $C \cdot (1 - f_l)^2$ |
| `KOZENY_CARMAN` | $C \cdot (1 - f_l)^2 / (f_l^3 + \varepsilon)$ |

### Теплопроводность на гранях (`KFaceMethod`)

`ARITHMETIC`, `HARMONIC`, `FROM_TEMP` (вычисление ступенчатой функции по температуре на грани).

## Структура проекта

```
src/
├── main.py                          # Пример точки входа
├── examples/
│   ├── stefan/                      # Чисто кондуктивная задача Стефана
│   ├── gallium/                     # Плавление галлия с конвекцией
│   ├── octadecane/                  # n-Октадекан в дифференциально нагреваемой полости
│   ├── water_convection/            # Естественная конвекция в жидкой воде
│   ├── water_freezing/              # Замерзание воды с конвекцией
│   ├── horizontal_layer/            # Плавление горизонтального слоя
│   ├── icicle/                      # Рост сосульки
│   ├── crevasse/                    # Таяние трещины
│   └── air/                         # Эталонный расчёт конвекции воздуха
├── parameters/
│   ├── config.py                    # ExperimentConfig (модель Pydantic, загрузка из JSON)
│   └── material_properties.py       # MaterialProperties (модель Pydantic)
├── core/
│   ├── geometry.py                  # DomainGeometry
│   ├── boundary_conditions.py       # BoundaryCondition, BoundaryConditions
│   ├── runner.py                    # SimulationState, ExperimentRunner
│   └── solvers/
│       ├── tridiagonal_solver.py    # Прогонка (алгоритм Томаса), Numba-JIT
│       └── mixins/adi.py            # ADIMixin (общая инфраструктура ADI-прогонок)
├── convective_operators/
│   ├── sf_based.py                  # Конвективные операторы на основе поля скорости
│   └── vorticity_based.py           # Якобиан (типа Аракавы) конвективного оператора
├── fluid_dynamics/
│   ├── utils.py                     # Помощники для вихря/скорости, миксин ГУ
│   └── solvers/
│       ├── solver_factory.py                     # IterativeNavierStokesSolver (классическая итерационная схема)
│       ├── bc_correction_solver_factory.py       # BCCorrectionNVSolver (схема с поправкой по ГУ, используется во всех расчётах)
│       ├── stream_function_solvers/
│       │   ├── amg.py               # Алгебраический многосеточный (PyAMG)
│       │   ├── cg.py                # Сопряжённые градиенты (CPU)
│       │   ├── cg_gpu.py            # Сопряжённые градиенты (GPU / CuPy)
│       │   ├── sor.py               # Верхняя релаксация (только классический Пуассон)
│       │   └── matrix_sweep.py      # Прямая прогонка (только классический Пуассон)
│       └── vorticity_solvers/
│           ├── peaceman_rachford.py  # ADI Писмена–Рэкфорда
│           ├── douglas_rachford.py   # ADI Дугласа–Рэкфорда
│           ├── loc_one_dim.py        # Локально-одномерная (LOD)
│           ├── vabishchevich.py      # Расщепление Вабищевича
│           ├── explicit.py           # Явная (Эйлера)
│           └── vab_fully_implicit.py # Полностью неявная схема Вабищевича
├── heat_transfer/
│   ├── coefficient_smoothing/
│   │   ├── coefficients.py          # Схемы ступенчатой/дельта-функций (erf, tanh, linear…)
│   │   └── mushy_zone.py            # Адаптивная оценка Δ по сетке
│   └── solvers/heat_transfer_solvers/
│       ├── peaceman_rachford.py      # ADI-решатель тепла Писмена–Рэкфорда
│       ├── douglas_rachford.py       # ADI-решатель тепла Дугласа–Рэкфорда
│       ├── loc_one_dim.py            # LOD-решатель тепла
│       ├── fully_implicit.py         # Полностью неявный решатель тепла
│       └── explicit.py              # Явный решатель тепла
└── utils/
    ├── boundary_conditions.py        # Фабрики ГУ Дирихле/Неймана
    └── nusselt.py                    # Вычисление числа Нуссельта
```

## Запуск тестов

```bash
pytest tests/
```
