import numpy as np

class Field(np.ndarray):
    def __new__(cls, arr, grid):
        obj = np.asarray(arr).view(cls)
        obj.grid = grid
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.grid = getattr(obj, 'grid', None)
        
    @property
    def shaped(self):
        if self.grid.dims:
            return self.reshape(self.grid.dims)
        return self

def make_circular_aperture(diameter, center=None):
    if center is None: shift = np.zeros(2)
    else: shift = np.array(center)
    
    def func(grid):
        x = grid.x - shift[0]
        y = grid.y - shift[1]
        f = (x**2 + y**2) <= (diameter / 2)**2
        return Field(f.astype('float'), grid)
    return func
