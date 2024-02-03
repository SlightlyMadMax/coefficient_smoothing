import scipy
import matplotlib.pyplot as plt
import numpy as np

fixed = np.load("../data/max_delta_testing/fixed_delta_depths_1500_240h_1m.npz")["depths"]
adaptive = np.load("../data/max_delta_testing/depths_1500_240h_1m.npz")["depths"]
times = [i for i in range(1, len(fixed) + 1)]
diff = [abs(x-y) for x, y in zip(fixed, adaptive)]
ax = plt.axes()
plt.plot(times, diff, linewidth=1, color='b')

plt.title("Разность глубин рассчитанных при фиксированной $\Delta$ и адаптивной")
ax.set_xlabel("Время, ч")
# ax.set_ylabel("$\Delta$, м")
ax.set_ylabel("Разность глубин, м")
# ax.legend()

plt.savefig(f"../graphs/delta/fixed_vs_adaptive.png")


# a = np.load("../data/max_delta_testing/adaptive_delta_1500_240h_1m.npz")["deltas"]
#
# print(a[-1])
# a_min = scipy.ndimage.minimum_filter(a, 4)
# a_max = scipy.ndimage.maximum_filter(a, 4)
#
# a = [0.5*(x+y) for (x, y) in zip(a_min, a_max)]
#
# ax = plt.axes()
# times_1 = [i for i in range(1, len(a)+1)]
# plt.plot(times_1, a, linewidth=1, color='b', label="1500x1500")
# plt.title("Зависимость $\Delta$ от времени")
# ax.set_xlabel("Время, ч")
# ax.set_ylabel("$\Delta$, м")
# # ax.set_ylabel("Глубина, м")
# ax.legend(title="Число узлов сетки")
#
# plt.savefig(f"../graphs/delta/deltas_sun_240h_avg.png")
