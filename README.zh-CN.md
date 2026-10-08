<p align="center"><img src="docs/assets/logo.png" width="72" alt="plot-function 标志"></p>

![plot-function — 从 NetCDF 到精美地图](docs/assets/header.png)

<p align="center"><a href="README.md">English</a> · <b>简体中文</b> · <a href="README.ja.md">日本語</a></p>
<p align="center">
  <a href="https://github.com/GISWLH/plot-function/actions/workflows/tests.yml"><img src="https://github.com/GISWLH/plot-function/actions/workflows/tests.yml/badge.svg" alt="测试"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square" alt="Python 3.10 及以上">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-78bda8?style=flat-square" alt="MIT 许可证"></a>
</p>

**只需一个 NetCDF 文件，几行代码即可绘制地图。** `plot-function` 将坐标处理、地图投影、色标和图片导出整合为简洁的 Python API 与命令行工具，同时保留原有的 **xarray → Cartopy → Matplotlib** 绘图思路，以及对图形对象的完整控制。

新工作流无需准备 Shapefile 或 GeoTIFF，也不依赖 Salem。可选海岸线由 Cartopy 的 Natural Earth 缓存提供；设置 `coastlines=False` 即可完全离线绘图。

[快速开始](#快速开始) · [示例画廊](#示例画廊) · [使用自己的数据](#使用自己的数据) · [命令行](#命令行) · [API 参考](docs/API.md)

## 快速开始

需要 **Python 3.10+**。请从仓库安装；以下步骤不依赖 PyPI 上是否已发布本项目。

```bash
git clone https://github.com/GISWLH/plot-function.git
cd plot-function
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell：.venv\Scripts\Activate.ps1
python -m pip install -e .
```

使用仓库自带的 NetCDF 数据绘制第一个月的气温：

```python
from plot_function import plot_map

result = plot_map(
    "data/ERA5temp_1978_monthly.nc",
    variable="t2m",
    isel={"time": 0},
    offset=-273.15, units="°C",  # 此文件中的温度单位为 K。
    cmap="RdYlBu_r",
    title="January 1978 · 2 m air temperature",
    output="january.png",
)
```

首次使用海岸线时可能下载 Natural Earth 数据。添加 `coastlines=False` 可避免下载。在 Notebook 中显示 `result.figure`；在具有图形界面的 Python 会话中调用 `matplotlib.pyplot.show()`。若标题使用中文，请自行配置支持中文的 Matplotlib 字体。

也可以运行一个完全离线、不依赖附带气候数据的轻量示例：

```bash
python examples/quickstart.py
```

脚本在 `examples/output/` 中生成明确标注为**合成数据**的 NetCDF 文件和 PNG 图片，仅需核心依赖。

## 示例画廊

以下四幅图均由 [`examples/gallery.py`](examples/gallery.py) 从仓库内的 **1978 年 ERA5 月平均 2 米气温 NetCDF 文件**生成，无需额外的矢量或栅格输入文件。

### 01 · 全球概览

Robinson 投影、简洁经纬网、分级温度配色与水平色标，适合展示整体空间分布。

![1978 年十二个月平均气温的算术平均地图](docs/assets/global-temperature.png)

### 02 · 相同色标下的季节对照

一月与七月使用完全一致的色标范围，便于可靠比较；深色主题适合演示文稿。

![1978 年一月和七月气温对照，共用色标](docs/assets/seasons.png)

### 03 · 季节差值

Equal Earth 投影配合以零为中心的对称色标，展示**七月减去一月**的温度差。这是同一年内的季节差异，不代表长期气候趋势或相对气候平均态的距平。

![1978 年七月减去一月的气温差](docs/assets/seasonal-contrast.png)

### 04 · 区域细节

使用 Lambert 正形圆锥投影展示东亚，并叠加带标签的等温线。同一个 NetCDF 文件即可支持全球和区域绘图。

<p align="center"><img src="docs/assets/east-asia.png" width="660" alt="1978 年七月东亚气温及等温线"></p>

重新生成画廊：

```bash
python examples/gallery.py
# 离线版本；另存输出，不覆盖仓库中的画廊：
python examples/gallery.py --no-coastlines --output examples/output/gallery
```

**计算说明：** 开尔文温度减去 273.15 转换为摄氏温度。全年图是十二个月平均值的等权算术平均，并非按月天数加权的年平均。经纬度每隔四个格点取一个，用于 1° 间距的显示；原始数据不变。海岸线使用 Natural Earth 110m 数据。详见[数据与美术素材说明](docs/assets/README.md)。

## 使用自己的数据

先查看变量、维度与单位：

```bash
plot-function inspect your-data.nc
```

```python
from plot_function import open_field, plot_map

# 按自己的文件修改变量名、维度名和层次值。
field = open_field(
    "your-data.nc", variable="temperature",
    isel={"time": 0}, sel={"level": 850},
)
result = plot_map(field, title="Temperature at 850 hPa", output="map.png")
```

对仓库内的数据求时间平均，再绘制区域图：

```python
result = plot_map(
    "data/ERA5temp_1978_monthly.nc", variable="t2m",
    reduce="time", statistic="mean",
    offset=-273.15, units="°C",
    projection="platecarree", extent=[90, 145, 5, 55],
    cmap="RdYlBu_r", title="East Asia · 1978 monthly-mean average",
)
result.save("figures/east-asia.pdf")
```

| 需求 | 参数 |
| :--- | :--- |
| 按位置 / 坐标值选择 | `isel={"time": 0}` / `sel={"level": 850}` |
| 沿额外维度聚合 | `reduce="time"`, `statistic="mean"` |
| 沿多个维度聚合 | `reduce=["time", "member"]` |
| 指定经纬度坐标名 | `latitude="nav_lat", longitude="nav_lon"` |
| 显式进行单位转换 | `scale=1, offset=-273.15, units="°C"` |
| 设置区域范围 | `extent=[西, 东, 南, 北]`，数值单位为度 |
| 多图保持可比 | 使用相同的 `levels` 或 `vmin` / `vmax` |
| 调整外观 | `theme="dark"`, `cmap="viridis"`, `plotfunc="contourf"` |
| 嵌入已有布局 | 传入 Cartopy `ax=`，使用该坐标轴的投影 |
| 导出图形 | `output="map.png"` 或 `result.save("map.svg")` |

投影名称支持 `robinson`、`platecarree`、`equalearth`、`mollweide`，也可直接传入 Cartopy 投影对象。返回的 `MapResult` 包含 `.figure`、`.axes`、`.artist`、`.colorbar`、`.data`，便于添加注释、共享色标和继续分析。批量绘图后调用 `plt.close(result.figure)` 释放图形。

### 数据支持范围

- **经纬度直角网格：** 经纬度坐标必须分别为一维、有限值，并各含至少两个不同点。支持常用 `lat`/`lon`、`latitude`/`longitude` 名称，以及 CF `standard_name` 或地理坐标 `units` 元数据。
- **科学处理显式指定：** 多个候选变量时需要 `variable=`；含多个值的非空间维度需要选择或聚合。统计不加权、忽略缺失值，支持 `mean`、`median`、`min`、`max`、`sum`、`std`。
- **坐标规范化：** 纬度排序，经度转换至 `[-180, 180)` 并排序；冗余的全球 0°/360° 端点会被移除，其他重复坐标会报错。
- **缺失值：** xarray 解码 NetCDF 填充值，非有限值会被掩膜；全为空的场会返回明确错误。
- **暂不直接支持：** 二维曲线网格、非结构网格、以米为单位的投影 x/y 坐标、跨日期变更线的区域网格或范围。这些数据需先预处理；本 API 不自动重投影源网格，也不猜测单位、垂直层次、时间权重或面积权重。
- **内存：** 选择或聚合后的二维场会载入内存，随后关闭文件。大型数据可先通过 xarray 处理，再传入 `DataArray`。

## 命令行

```bash
plot-function plot data/ERA5temp_1978_monthly.nc \
  --variable t2m --isel '{"time": 0}' \
  --offset -273.15 --units '°C' --cmap RdYlBu_r \
  --title 'January 1978' --output january.png

# 无图形界面、完全离线的区域月平均值汇总图：
plot-function plot data/ERA5temp_1978_monthly.nc \
  --variable t2m --reduce time --projection platecarree \
  --extent 90 145 5 55 --no-coastlines --output regional.png
```

`python -m plot_function` 与 `plot-function` 等价。运行 `plot-function plot --help` 查看所有参数。CLI 使用 Matplotlib 的 Agg 后端，无需桌面环境。选择参数使用 JSON 对象，请按所用 Shell 正确加引号。

## 兼容已有 Notebook

```python
from utils import plot              # 原导入方式继续可用。
from plot_function import legacy   # 新包中的同一组底层函数。
```

保留原有地图、阴影、区域与升温情景面板绘图函数，并修复海岸线关键字参数传递、阴影反转及返回值、区域范围、图例、色标覆盖参数和剖面标签等问题。详见[迁移说明](docs/MIGRATION.md)。

历史 Notebook 所需工具作为可选依赖提供：

```bash
python -m pip install -e '.[notebook,legacy]'
jupyter lab
```

[`plotbook.ipynb`](plotbook.ipynb) 保留了原有栅格和 Shapefile 示例。其中中国气温部分引用未随仓库提供的 `data/tp/tmp_2022.nc`，因此不能直接完整执行。旧版中国边界函数依赖仓库 `data/` 中的文件，或显式提供的 Shapefile 路径；大型示例数据不打包进 Python wheel。新用户建议从上面的 NetCDF 示例开始。

## 开发与贡献

```bash
python -m pip install -e '.[dev]'
pytest
ruff check plot_function utils tests examples
python examples/quickstart.py
python -m build
```

测试覆盖 NetCDF 选择、坐标与单位处理、无网络绘图、CLI，以及旧接口回归问题。GitHub Actions 配置为在 Python 3.10 和 3.12 上运行测试。欢迎贡献，参见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 致谢与许可证

作者：**Longhao Wang**。基于 [xarray](https://docs.xarray.dev/)、[Cartopy](https://scitools.org.uk/cartopy/docs/latest/)、[Matplotlib](https://matplotlib.org/) 和 [mplotutils](https://github.com/mathause/mplotutils) 构建。软件采用 [MIT 许可证](LICENSE)。数据来源、Natural Earth 署名，以及装饰性美术素材与真实数据绘图的区分，见[素材说明](docs/assets/README.md)。
