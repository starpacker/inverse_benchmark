import numpy as np

class Grid(object):
    def __init__(self, coords, weights=None):
        self.coords = coords
        self.weights = weights
        self._input_grid = None
        
    @property
    def x(self):
        return self.coords[0]
    
    @property
    def y(self):
        return self.coords[1]
        
    @property
    def size(self):
        return self.coords[0].size
        
    @property
    def ndim(self):
        return len(self.coords)
    
    def zeros(self, dtype=None):
        return Field(np.zeros(self.size, dtype=dtype), self)
        
    def ones(self, dtype=None):
        return Field(np.ones(self.size, dtype=dtype), self)
        
    def scaled(self, scale):
        new_coords = [c * scale for c in self.coords]
        new_weights = self.weights * (scale**2) if self.weights is not None else None
        return CartesianGrid(new_coords, new_weights)
        
    def shifted(self, shift):
        new_coords = [c + s for c, s in zip(self.coords, shift)]
        return CartesianGrid(new_coords, self.weights)
        
    def subset(self, mask):
        new_coords = [c[mask] for c in self.coords]
        new_weights = self.weights[mask] if self.weights is not None else None
        return Grid(new_coords, new_weights)

class CartesianGrid(Grid):
    def __init__(self, coords, weights=None):
        super().__init__(coords, weights)
        self.is_regular = True 
        try:
            self.dims = [int(np.sqrt(self.size))] * 2
            self.delta = [coords[0][1] - coords[0][0], coords[1][self.dims[0]] - coords[1][0]]
        except:
            self.dims = None
            self.delta = None

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

def make_uniform_grid(dims, extent):
    if np.isscalar(dims): dims = [dims, dims]
    if np.isscalar(extent): extent = [extent, extent]
    
    x = np.linspace(-extent[0]/2, extent[0]/2, dims[0], endpoint=False) + extent[0]/(2*dims[0])
    y = np.linspace(-extent[1]/2, extent[1]/2, dims[1], endpoint=False) + extent[1]/(2*dims[1])
    
    xx, yy = np.meshgrid(x, y)
    coords = [xx.ravel(), yy.ravel()]
    weights = (extent[0]/dims[0]) * (extent[1]/dims[1])
    
    grid = CartesianGrid(coords, weights)
    grid.dims = dims
    grid.delta = [extent[0]/dims[0], extent[1]/dims[1]]
    return grid
