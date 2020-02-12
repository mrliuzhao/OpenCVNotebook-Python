from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['Microsoft YaHei']

# x = np.arange(0, 11)
x = np.arange(0, 3 * np.pi, 0.1)
# y = 2 * x + 5
y = np.sin(x)
plt.title("Matplotlib demo - 中文测试")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x, y, '--g')
plt.show()



