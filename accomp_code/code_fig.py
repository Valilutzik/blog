from mpl_toolkits.axes_grid.axislines import SubplotZero
from matplotlib.transforms import BlendedGenericTransform
import matplotlib.pyplot as plt
import numpy

if 1:
    fig = plt.figure(1)
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)

    ax.axhline(linewidth=1.7, color="black")
    ax.axvline(linewidth=1.7, color="black")

    plt.xticks(range(5))
    plt.yticks(range(1,5))

    ax.text(0, 1.05, '$x_{2}$', transform=BlendedGenericTransform(ax.transData, ax.transAxes), ha='center')
    ax.text(1.025, 0, '$x_{1}$', transform=BlendedGenericTransform(ax.transAxes, ax.transData), va='center')

    for direction in ["xzero", "yzero"]:
        ax.axis[direction].set_axisline_style("-|>")
        ax.axis[direction].set_visible(True)

    for direction in ["left", "right", "bottom", "top"]:
        ax.axis[direction].set_visible(False)

    x = numpy.linspace(-1, 3.5, 1000)
    y = (4 - 2*x)
    ax.plot(x,y)

    plt.annotate('$y=1$',xy=(1.75,1.5))
    plt.annotate('$y=0$',xy=(0.5,1))

    plt.show()