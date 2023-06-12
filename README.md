Envi


```python
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.hatch
import matplotlib.pyplot as plt
import mplotutils as mpu
import numpy as np
from matplotlib.path import Path
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
```

# Function List

Please see the plot.py

# Demo

## Global IPCC


```python
from utils import plot
```


```python
import xarray as xr
file_name='data/ERA5temp_1978_monthly.nc' 
ds=xr.open_dataset(file_name)
lat = ds['latitude']
lon = ds['longitude']


ds = ds.rename_dims({'latitude':'lat','longitude':'lon'})
ds.coords['lat'] = ('lat', lat.to_numpy())
ds.coords['lon'] = ('lon', lon.to_numpy()) # 对维度lon指定新的坐标信息lon
ds = ds.reset_coords(names=['latitude','longitude'], drop=True)
ds['t2m'] = ds['t2m'] - 273.15
ds['t2m']
```




```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
fig = plt.figure()
proj = ccrs.PlateCarree() #ccrs.Robinson()ccrs.Mollweide()Mollweide()
ax = fig.add_subplot(111, projection=proj)
levels = np.linspace(-30, 30, num=19)
plot.one_map_flat(ds['t2m'][0], ax, levels=levels, cmap="BrBG_r", mask_ocean=False, add_coastlines=True, add_land=False, plotfunc="pcolormesh")
```




​    
![png](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/output_9_1.png)
​    



```python
#import rioxarray as xrx
#p = rxr.open_rasterio(filename)
p = np.mean(ds['t2m'], 0) > -20
```


```python
fig = plt.figure()  
proj = ccrs.PlateCarree()  #ccrs.Robinson()  
#proj = ccrs.Robinson()  
ax = fig.add_subplot(111, projection=proj)
levels = np.linspace(-30, 30, num=19)
plot.one_map(ds['t2m'], ax,  average='mean', dim='time', cmap="RdBu_r", levels=levels,  mask_ocean=True,  add_coastlines=True,  add_land=True,  plotfunc="pcolormesh", colorbar=True, getmean=True)
plot.hatch_map(ax, p, 3 * ".", label="Lack of model agreement", invert=True, linewidth=0.25, color="0.1")
```


​    
![png](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/output_11_1.png)
​    



```python
at_warming_c = []
at_warming_c.append(ds['t2m'][5:8])
at_warming_c.append(ds['t2m'][9:12])
at_warming_c.append(ds['t2m'][0:3])
len(at_warming_c)

#fig = plt.figure()   
#proj = ccrs.Robinson()  
#
#ax = fig.add_subplot(131, projection=proj)
plot.at_warming_level_one(at_warming_c=at_warming_c, unit="Change (times as frequent)", title='drought frequency change w.r.t. 1850-1900', \
                     average="median",  mask_ocean=True,  colorbar=True, cmap="RdBu",  dim='time', add_legend=False, hatch_data=None, levels=levels, plotfunc='pcolormesh', getmean=True)

```


​    
![png](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/output_12_1.png)
​    


## Regional


```python
import xarray as xr
file_name='data/ET_trend.tif' 
ds=xr.open_dataset(file_name)
data = ds['band_data'][0]
```


```python
file_name='data/ET_p.tif' 
ds=xr.open_dataset(file_name)
p = ds['band_data'][0]
```


```python
datap = p < 0.05
lat = ds['y']
lon = ds['x']
datap = datap.swap_dims({'y':'lat','x':'lon'})
datap.coords['lat'] = ('lat',lat.to_numpy())
datap.coords['lon'] = ('lon',lon.to_numpy()) # 对维度lon指定新的坐标信息lon
datap = datap.reset_coords(names=['y','x'], drop=True)
```


```python
datap
```


```python
import salem
import geopandas as gpd
shp_dir='data/Yangtze_4326.shp'
shpfile=gpd.read_file(shp_dir)
pmaskregion=datap.salem.roi(shape=shpfile)
```

区域一般选择默认投影，因此不能修改投影


```python
from utils import plot
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
fig = plt.figure()
proj = ccrs.PlateCarree() #ccrs.Robinson()ccrs.Mollweide()Mollweide()
ax = fig.add_subplot(111, projection=proj)
levels = np.array([-1, -0.8, -0.4, 0, 0.4, 0.8, 1])#levels = np.linspace(-1, 1, num=19)
plot.one_map_region(data, ax, levels=levels,  extents=[89, 125, 23, 37], interval=[9, 7], mask_ocean=False, add_coastlines=False, add_land=False, add_gridlines=True, colorbar=True, plotfunc="pcolormesh")
plot.hatch_map(ax, pmaskregion, 3 * "/", label="Lack of model agreement", invert=True, linewidth=0.25, color="0.1")
#plt.savefig("test.png", dpi=300)
```


​    
![png](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/output_20_1.png)
​    


在新的``plot.one_map_region``函数中，绘制全球格网经纬度地图，需要指定范围和间隔，而且不能改投影，不太方便


```python
import xarray as xr
file_name='data/ERA5temp_1978_monthly.nc' 
ds=xr.open_dataset(file_name)
lat = ds['latitude']
lon = ds['longitude']


ds = ds.rename_dims({'latitude':'lat','longitude':'lon'})
ds.coords['lat'] = ('lat', lat.to_numpy())
ds.coords['lon'] = ('lon', lon.to_numpy()) # 对维度lon指定新的坐标信息lon
ds = ds.reset_coords(names=['latitude','longitude'], drop=True)
ds['t2m'] = ds['t2m'] - 273.15
ds['t2m']
```




```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
fig = plt.figure()
proj = ccrs.PlateCarree() #ccrs.Robinson()ccrs.Mollweide()Mollweide()
ax = fig.add_subplot(111, projection=proj)
levels = np.linspace(-30, 30, num=19)
plot.one_map_region(ds['t2m'][0], ax, extents=[-180, 180, -90, 90], interval=[60, 30], levels=levels, cmap="RdBu", mask_ocean=False, add_coastlines=True, add_land=False, add_gridlines=True, colorbar=True, plotfunc="pcolormesh")
```


![png](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/output_23_1.png)
    


因此，有新的``one_map_global_line``函数，默认了格网经纬度


```python
import xarray as xr
file_name='data/r2.tif' 
ds=xr.open_dataset(file_name)
data = ds['band_data'][0]
```


```python
from utils import plot
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
fig = plt.figure()
proj = ccrs.Robinson() #ccrs.Robinson()ccrs.Mollweide()Mollweide()
ax = fig.add_subplot(111, projection=proj)
levels = np.linspace(0, 1, num=9)
plot.one_map_global_line(data, ax, levels=levels, cmap="RdBu",  mask_ocean=False, add_coastlines=True, add_land=True,  colorbar=True, plotfunc="pcolormesh")
```


​    
![png](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/output_26_1.png)
​    



```python
from utils import plot
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
fig = plt.figure()
proj = ccrs.Robinson() #ccrs.Robinson()ccrs.Mollweide()Mollweide()
ax = fig.add_subplot(111, projection=proj)
levels = np.linspace(0, 1, num=9)
plot.one_map_flat(data, ax, levels=levels, cmap="RdBu",  mask_ocean=False, add_coastlines=True, add_land=True,  colorbar=True, plotfunc="pcolormesh")

```


​    
![png](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/output_27_1.png)
​    


由于区域尺度特殊性，不容易看出区域的位置，因此又添加了``add_river``,``add_lake``,``add_stock``函数

上述函数分别决定是否添加河流、湖泊和背景影像图


```python
import xarray as xr
file_name='data/data_trend.tif' 
ds=xr.open_dataset(file_name)
data = ds['band_data'][0]
```


```python
file_name='data/data_p.tif' 
ds=xr.open_dataset(file_name)
p = ds['band_data'][0]
```


```python
datap = p < 0.05
lat = ds['y']
lon = ds['x']
datap = datap.swap_dims({'y':'lat','x':'lon'})
datap.coords['lat'] = ('lat',lat.to_numpy())
datap.coords['lon'] = ('lon',lon.to_numpy()) # 对维度lon指定新的坐标信息lon
datap = datap.reset_coords(names=['y','x'], drop=True)
```


```python
import salem
import geopandas as gpd
shp_dir='data/GRDC.shp'
shpfile=gpd.read_file(shp_dir)
pmaskregion=datap.salem.roi(shape=shpfile)
```


```python
from utils import plot
```


```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cmaps
fig = plt.figure()
proj = ccrs.PlateCarree() #ccrs.Robinson()ccrs.Mollweide()Mollweide()
ax = fig.add_subplot(111, projection=proj)
levels = np.array([-10, -5, -3, -1, 0, 1, 3, 5, 10])#levels = np.linspace(-1, 1, num=19)
plot.one_map_region(data, ax, cmap=cmaps.temp_19lev_r, levels=levels,  extents=[30, 130, 22, 58], interval=[20, 18], mask_ocean=False, add_coastlines=True, add_land=False, add_river=True, add_lake=True, add_stock=True, add_gridlines=True, colorbar=True, plotfunc="pcolormesh")
plot.hatch_map(ax, pmaskregion, 3 * "/", label="Lack of model agreement", invert=True, linewidth=0.25, color="black")
#cmap='RdYlGn' colors=mycolor
#plt.savefig('temp.png', dpi=300) 
```


![png](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/output_36_1.png)
    


## China
    


中国


```python
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.hatch
import matplotlib.pyplot as plt
import mplotutils as mpu
import numpy as np
from matplotlib.path import Path
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
```


```python
import xarray as xr
import numpy as np
file_name = 'D:/Onedrive/data/tp/tmp_2022.nc' 
ds=xr.open_dataset(file_name)
da = np.mean(ds['tmp'], 0) * 0.1
```


```python
import salem
import geopandas as gpd
shp_dir='data/china.shp'
shpfile=gpd.read_file(shp_dir)
damask=da.salem.roi(shape=shpfile)
```


```python
from utils import plot
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cmaps
fig = plt.figure()
#proj = ccrs.PlateCarree() #ccrs.Robinson()ccrs.Mollweide()Mollweide()
proj = ccrs.LambertConformal(central_longitude=105, 
                              central_latitude=40,
                              standard_parallels=(25.0, 47.0))
ax = fig.add_subplot(111, projection=proj)
levels = np.array([-6, -4, -2, 0, 2, 4, 8, 15, 20, 25])#levels = np.linspace(-1, 1, num=19)
plot.one_map_china(damask, ax, cmap=cmaps.temp_19lev, levels=levels,  mask_ocean=False, add_coastlines=True, add_land=False, add_river=True, add_lake=True, add_stock=False, add_gridlines=True, colorbar=True, plotfunc="pcolormesh")
ax2 = fig.add_axes([0.708, 0.174, 0.2, 0.3], projection = proj)
plot.sub_china_map(damask, ax2,  add_coastlines=True, add_land=False)
```


​    
![png](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/output_69_1.png)
​    



```python
pa = da > -2
pamask=pa.salem.roi(shape=shpfile)
```


```python
from utils import plot
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cmaps
fig = plt.figure()
proj = ccrs.PlateCarree() #ccrs.Robinson()ccrs.Mollweide()Mollweide()
ax = fig.add_subplot(111, projection=proj)
levels = np.array([-6, -4, -2, 0, 2, 4, 8, 15, 20, 25])#levels = np.linspace(-1, 1, num=19)
plot.one_map_china(damask, ax, cmap=cmaps.temp_19lev, levels=levels,  mask_ocean=False, add_coastlines=True, add_land=False, add_river=True, add_lake=True, add_stock=True, add_gridlines=True, colorbar=True, plotfunc="pcolormesh")
plot.hatch_map(ax, pamask, 3 * "/", label="Lack of model agreement", invert=False, linewidth=0.25, color="black")
ax2 = fig.add_axes([0.71, 0.196, 0.2, 0.3], projection = proj)
plot.sub_china_map(damask, ax2,  add_coastlines=True, add_land=False, add_stock=True)
plot.hatch_map(ax2, pamask, 3 * "/", label="Lack of model agreement", invert=False, linewidth=0.25, color="black")
```


​    
![png](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/output_71_1.png)
​    
