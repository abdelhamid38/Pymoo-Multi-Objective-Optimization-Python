# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from deap import benchmarks
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import numpy as np
import plotly.graph_objects as go

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)

x, y = np.meshgrid(x, y)
z1 = np.zeros(x.shape)
z2 = np.zeros(y.shape)

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        z1[i, j], z2[i, j] = benchmarks.kursawe((x[i, j], y[i, j]))

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(x, y, z1, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)

plt.xlabel("X")
plt.ylabel("Y")

plt.title("KURSAWE f1")

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(x, y, z2, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)

plt.xlabel("X")
plt.ylabel("Y")

plt.title("KURSAWE f2")

plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
plt.show()


class ProblemWrapper(Problem):
    def _evaluate(self, designs, out, *args, **kwargs):
        res = []
        for design in designs:
            res.append(benchmarks.kursawe(design))

        out['F'] = np.array(res)


problem = ProblemWrapper(n_var=2, n_obj=2, xl=[-5., -5.], xu=[5., 5.])

algorithm = NSGA2(pop_size=100)
stop_criteria = ('n_gen', 100)
results = minimize(
    problem=problem,
    algorithm=algorithm,
    termination=stop_criteria
)

var = results.F
res_data = results.F.T
fig = go.Figure(data=go.Scatter(x=res_data[0], y=res_data[1], mode='markers'))
fig.show()
