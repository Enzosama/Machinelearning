from matplotlib.axes import Axes
from matplotlib.contour import ContourSet

class TriContourSet(ContourSet):
    def __init__(self, ax: Axes, *args, **kwargs) -> None: ...

def tricontour(ax: Axes, *args, **kwargs) -> TriContourSet: ...
def tricontourf(ax: Axes, *args, **kwargs) -> TriContourSet: ...
