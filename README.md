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
type(np.array(ds['t2m'][0]))
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
![png](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/output_3_1.png)
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
![png](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/output_5_1.png)
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


  
![png](https://imagecollection.oss-cn-beijing.aliyuncs.com/legion/output_6_1.png)
    

