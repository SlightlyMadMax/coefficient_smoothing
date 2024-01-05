import matplotlib.pyplot as plt
import numpy as np

a = np.load("../data/max_delta_testing/depths_1500_72h_1m_sun.npz")["depths"]

print(a[-1])

ax = plt.axes()
times_1 = [i / 30 for i in range(1, len(a)+1)]
plt.plot(times_1, a, linewidth=1, color='b', label="1500x1500")
plt.title("Изменение глубины трещины")
ax.set_xlabel("Время, ч")
ax.set_ylabel("$\Delta$, м")
ax.legend(title="Число узлов сетки")

plt.savefig(f"../graphs/delta/depth_sun.png")
