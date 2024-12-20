import math

# Коэффициент теплообмена льда
CONV_COEF = 25.0

A = 15.4

B = -3806.9

WIND_SPEED = 10.0

REL_HUMIDITY = 0.5

CLOUDINESS = 0.5

# Солнечная постоянная
Q_SOL = 1360.0

# Географическая широта
LAT = -69.0 * math.pi / 180.0

# Амплитуда склонения солнца
DECL = 23.5 * math.pi / 180.0

# Угловая скорость вращения Земли
RAD_SPEED = 7.292 / 100000.0

# НАЧАЛЬНАЯ ТЕМПЕРАТУРА ВОЗДУХА
T_air = 257.15

# АМПЛИТУДА ИЗМЕНЕНИЯ ТЕМПЕРАТУРЫ ВОЗДУХА (СУТОЧНАЯ)
T_amp_day = 8.0

# АМПЛИТУДА ИЗМЕНЕНИЯ ТЕМПЕРАТУРЫ ВОЗДУХА (ГОДЧИНАЯ)
T_amp_year = 8.0

# Gravitational Acceleration
G = 9.81

# Absolute Zero Temperature
ABS_ZERO = -273.15

# ДЛЯ ТЕСТОВ
# N_X = 21
# WIDTH = 1.0
# dx = WIDTH / (N_X - 1)
# inv_dx = (N_X - 1) / WIDTH
# N_Y = 1001
# HEIGHT = 8.0
# dy = HEIGHT / (N_Y - 1)
# inv_dy = (N_Y - 1) / HEIGHT
# WATER_H = 0.05
# CREV_DEPTH = 0.8
# delta = 0.3
# T_ICE_MIN = -5.0
# T_WATER_MAX = 5.0
# T_0 = 0.0
# dt = 3600.0
# FULL_TIME = 60.0 * 60.0 * 24.0 * 300.0
# N_T = int(FULL_TIME / dt)
# C_WATER = 4120.7
# C_ICE = 2056.8
# RHO_WATER = 999.84
# RHO_ICE = 918.9
# C_WATER_VOL = C_WATER * RHO_WATER
# C_ICE_VOL = C_ICE * RHO_ICE
# L = 333000.0
# L_VOL = L * RHO_WATER
# K_WATER = 0.59
# K_ICE = 2.21
# CONV_COEF = 6000.0
# Q_SOL = 1360.0
# LAT = -69.0 * math.pi / 180.0
# DECL = -23.0 * math.pi / 180.0
# RAD_SPEED = 7.292 / 100000.0
# T_air = 275.65
# T_amp = 2.5
