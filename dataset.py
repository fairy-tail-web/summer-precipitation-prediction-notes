import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
import os
import warnings
import cv2
import pandas as pd
import logging
import glob
import rasterio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
MONTHLY_IMF_FEATURES = {
    6: [  'nino34_imf3'], # For June
    7: ['pdo_imf4'],                                              # For July
    8: [ 'nino34_imf3']        # For August
}
# MONTHLY_IMF_FEATURES = {
#     6: ['nino34_imf4', 'nino34_imf3', 'pmm_imf7', 'nino34_imf8', 'pdo_imf9'], # For June
#     7: ['pdo_imf4', 'pmm_imf1'],                                              # For July
#     8: ['nino34_imf3', 'pdo_imf1', 'pmm_imf3', 'pmm_imf2', 'pmm_imf4']        # For August
# }
class tpDataset_multi(Dataset):
    """
    一个为降水预测任务定制的数据集类。
    ✨✨✨ 核心修正 v12 (集成多尺度IMFs):
    - 新增DEM高程作为静态空间特征。
    - 解决了DEM与大气场空间范围不匹配的问题。
    - ✨✨✨ 新增：加载来自EEMD分解的“多尺度”气候指数IMFs。
    """

    def __init__(self, type_, data_dir, shuffle_flag, target_month,
                 dem_path,
                 climate_index_path,  # (nino34_monthly.txt)
                 climate_imf_path,  # ✨✨✨ 新增: 指向 climate_indices_imfs.csv
                 input_len=3, output_len=1, target_size=(32, 64),
                 target_lat_range=None, target_lon_range=None):
        super().__init__()
        assert type_ in ['train', 'val', 'test']
        self.type = type_
        self.data_path = os.path.join(data_dir, self.type)
        self.shuffle_flag = shuffle_flag
        self.target_h, self.target_w = target_size
        self.target_month = target_month
        self.input_len = input_len
        self.output_len = output_len
        self.total_seq_len = self.input_len + self.output_len
        self.required_months = list(range(3, 9))  # 您的数据只包含3-8月
        self.feature_names = []

        self.dem_path = dem_path
        self.target_lat_range = target_lat_range
        self.target_lon_range = target_lon_range
        self.lat_slice = None
        self.lon_slice = None

        climatology_map_path = os.path.join(data_dir, 'sst_climatology_map.nc')
        if not os.path.exists(climatology_map_path):
            raise FileNotFoundError(f"SST气候态【地图】文件未找到: {climatology_map_path}")
        self.sst_climatology_map = xr.open_dataset(climatology_map_path)
        logger.info(f"SST气候态地图加载成功: {climatology_map_path}")

        # (加载 nino34_monthly.txt)
        try:
            logger.info(f"正在加载月度气候指数文件: {climate_index_path}")
            # ✨ 修复: 使用 sep='\s+'
            self.climate_index_df = pd.read_csv(climate_index_path, sep='\s+', skiprows=1,
                                                names=['YR', 'MON', 'NINO1+2', 'ANOM1+2', 'NINO3', 'ANOM3', 'NINO4',
                                                       'ANOM4', 'NINO3.4', 'ANOM3.4'])
            self.climate_index_df.set_index(['YR', 'MON'], inplace=True)
            logger.info("月度气候指数数据加载并索引成功。")
        except FileNotFoundError:
            logger.error(f"气候指数文件未找到: {climate_index_path}")
            raise
        except Exception as e:
            logger.error(f"加载气候指数文件时出错: {e}", exc_info=True)
            raise

        # 加载 EEMD 分解后的 IMFs 文件 ---
        try:
            logger.info(f"正在加载预分解的气候指数IMFs: {climate_imf_path}")
            # 加载 'climate_indices_imfs.csv'，并将 'date' 列解析为日期时间对象
            self.climate_imf_df_all = pd.read_csv(climate_imf_path, parse_dates=['date'])  # <--- 改为 _all
            # 将 'date' 列设为索引，以便于后续按时间戳查找
            self.climate_imf_df_all.set_index('date', inplace=True)
            logger.info("气候指数IMFs加载并索引成功。")
        except FileNotFoundError:
            logger.error(f"气候指数IMFs文件未找到: {climate_imf_path}")
            raise
        except Exception as e:
            logger.error(f"加载气候指数IMFs文件时出错: {e}", exc_info=True)
            raise

        logger.info(f"\n{'=' * 20} 正在为时空模型初始化 '{type_}' 数据集 {'=' * 20}")
        self._prepare_data()

    def _load_data_from_files(self):
        def standardize_time_coord(ds):
            return ds.rename({'time': 'valid_time'}) if 'time' in ds.coords and 'valid_time' not in ds.coords else ds

        logger.info(f"\n--- 正在加载 '{self.type}' 集的主要 .nc 文件 ---")
        main_data_files = sorted(glob.glob(os.path.join(self.data_path, "data_*.nc")))
        if not main_data_files: raise FileNotFoundError(f"在 {self.data_path} 中找不到 'data_*.nc' 文件。")

        main_ds = xr.open_mfdataset(main_data_files, combine='by_coords', parallel=False,
                                    preprocess=standardize_time_coord)

        logger.info("正在清理主要数据的时间坐标，确保唯一性...")
        _, unique_indices = np.unique(main_ds['valid_time'].values, return_index=True)
        main_ds = main_ds.isel(valid_time=unique_indices)

        if self.target_lat_range and self.target_lon_range:
            logger.info(f"正在将大气场裁剪到范围: LAT {self.target_lat_range}, LON {self.target_lon_range}")
            main_ds = main_ds.where(
                (main_ds.latitude >= self.target_lat_range[0]) & (main_ds.latitude <= self.target_lat_range[1]) &
                (main_ds.longitude >= self.target_lon_range[0]) & (main_ds.longitude <= self.target_lon_range[1]),
                drop=True
            )

        self.time_coords = main_ds['valid_time'].values
        all_feature_arrays = []

        # ... (大气场, DEM, SST 的加载逻辑完全保持不变) ...
        atm_vars = ['q', 'r', 't', 'u', 'v', 'w', 'z']
        sfc_vars = ['tp']
        pressure_levels = [850, 700, 500, 200]

        logger.info("--- 正在提取标准大气和地表特征 ---")
        # ... (此 for 循环保持不变) ...
        for var in sfc_vars + atm_vars:
            if var in main_ds:
                variable_data = main_ds[var]
                if var == 'tp':
                    logger.info("对 'tp' 变量应用 log1p 变换...")
                    tp_values = variable_data.values
                    processed_data = np.log1p(tp_values)
                    all_feature_arrays.append(np.expand_dims(processed_data, axis=1))
                    self.feature_names.append(var)
                elif 'pressure_level' in variable_data.dims:
                    if var == 'z':
                        variable_data = variable_data / 9.80665
                    for level in pressure_levels:
                        if level in variable_data.coords.get('pressure_level', []):
                            all_feature_arrays.append(
                                np.expand_dims(variable_data.sel(pressure_level=level).values, axis=1))
                            self.feature_names.append(f'{var}_{level}')
                else:
                    processed_data = variable_data.values
                    if var == 'z':
                        processed_data = processed_data / 9.80665
                    all_feature_arrays.append(np.expand_dims(processed_data, axis=1))
                    self.feature_names.append(var)

        # ... (DEM 加载逻辑保持不变) ...
        logger.info("--- 正在加载和处理DEM高程数据 ---")
        with rasterio.open(self.dem_path) as dem_src:
            dem_data = dem_src.read(1)  # 读取第一个波段
            dem_bounds = dem_src.bounds
            # ... (DEM 的 'large_dem_canvas' 逻辑不变) ...
            large_dem_canvas = np.zeros(main_ds['tp'].shape[-2:], dtype=np.float32)
            main_lats = main_ds.latitude.values
            main_lons = main_ds.longitude.values
            lat_indices = np.where((main_lats >= dem_bounds.bottom) & (main_lats <= dem_bounds.top))[0]
            lon_indices = np.where((main_lons >= dem_bounds.left) & (main_lons <= dem_bounds.right))[0]

            if len(lat_indices) > 0 and len(lon_indices) > 0:
                lat_start, lat_end = lat_indices.min(), lat_indices.max()
                lon_start, lon_end = lon_indices.min(), lon_indices.max()
                target_pixel_h = lat_end - lat_start + 1
                target_pixel_w = lon_end - lon_start + 1
                resized_dem = cv2.resize(dem_data, (target_pixel_w, target_pixel_h), interpolation=cv2.INTER_CUBIC)
                large_dem_canvas[lat_start:lat_end + 1, lon_start:lon_end + 1] = resized_dem
                dem_broadcasted = np.broadcast_to(large_dem_canvas, (len(self.time_coords),) + large_dem_canvas.shape)
                all_feature_arrays.append(np.expand_dims(dem_broadcasted, axis=1))
                self.feature_names.append('dem')
                logger.info("DEM数据已成功处理并添加为静态特征。")
            else:
                logger.warning("DEM地理范围与大气场数据没有重叠，跳过DEM特征。")

        # ... (SST 特征处理逻辑保持不变) ...
        logger.info("--- 正在加载SST全图并计算空间距平特征 ---")
        sst_map_filepath = os.path.join(self.data_path, 'sst_full.nc')
        if os.path.exists(sst_map_filepath):
            try:
                with xr.open_dataset(sst_map_filepath).pipe(standardize_time_coord) as sst_map_ds:
                    _, unique_indices_sst = np.unique(sst_map_ds['valid_time'].values, return_index=True)
                    sst_map_ds = sst_map_ds.isel(valid_time=unique_indices_sst)
                    sst_anomaly_map = sst_map_ds['sst'].groupby('valid_time.month') - self.sst_climatology_map['sst']
                    aligned_anomaly_map = sst_anomaly_map.reindex(valid_time=main_ds.valid_time, method='nearest')
                    regridded_anomaly_map = aligned_anomaly_map.interp_like(main_ds)
                    all_feature_arrays.append(np.expand_dims(np.nan_to_num(regridded_anomaly_map.values), axis=1))
                    self.feature_names.append('sst_anomaly_map')
                    logger.info("成功计算并添加了 'sst_anomaly_map' 空间特征。")
            except Exception as e:
                logger.error(f"处理完整SST地图时出错: {e}", exc_info=True)
                raise e

        self.data = np.concatenate(all_feature_arrays, axis=1)
        H, W = self.data.shape[2], self.data.shape[3]  # 获取 H 和 W
        time_stamps = pd.to_datetime(self.time_coords)  # 获取时间戳

        # (Niño 3.4 加载逻辑保持不变)
        logger.info("--- 正在整合大尺度气候指数特征 (Niño 3.4) ---")
        nino34_feature_list = []
        for ts in time_stamps:
            year, month = ts.year, ts.month
            try:
                nino_value = self.climate_index_df.loc[(year, month), 'ANOM3.4']
                nino34_feature_list.append(nino_value)
            except KeyError:
                # 您的数据从1948年开始，nino34从1950年开始，这里会触发
                logger.warning(
                    f"在 nino34_monthly.txt 中找不到年份={year}, 月份={month}的数据。将使用0.0 (气候中性态) 进行填补。")
                nino34_feature_list.append(0.0)

        nino34_feature_array = np.array(nino34_feature_list, dtype=np.float32)
        nino34_broadcastable = nino34_feature_array.reshape(-1, 1, 1, 1)
        nino34_full_feature = np.broadcast_to(nino34_broadcastable, (len(self.time_coords), 1, H, W))
        self.data = np.concatenate([self.data, nino34_full_feature], axis=1)
        self.feature_names.append('nino34')
        logger.info("成功添加 'nino34' 作为全局特征。")

        logger.info(f"--- 正在为目标月份 {self.target_month} 整合特定的 IMF 特征 ---")

        # 从全局字典获取当前月份所需的IMF列表
        imf_features_to_add = MONTHLY_IMF_FEATURES.get(self.target_month, [])  # 使用 .get() 提供空列表作为默认值

        if not imf_features_to_add:
            logger.warning(f"未找到目标月份 {self.target_month} 的IMF特征列表，将不添加任何IMF特征。")
        else:
            logger.info(f"将要添加的IMF特征: {imf_features_to_add}")

            for feature_name in imf_features_to_add:
                if feature_name not in self.climate_imf_df_all.columns:  # 检查是否存在于加载的总IMF DataFrame中
                    logger.warning(f"在 climate_indices_imfs.csv 中找不到特征 '{feature_name}'，跳过。")
                    continue

                feature_list = []
                for ts in time_stamps:
                    try:
                        # ✨ 使用标准化时间戳查找
                        normalized_ts = ts.normalize()
                        value = self.climate_imf_df_all.loc[normalized_ts, feature_name]
                        feature_list.append(value)
                    except KeyError:
                        normalized_ts_for_error = ts.normalize()
                        logger.error(
                            f"在IMFs文件中找不到 {feature_name} 的标准化日期 {normalized_ts_for_error} (原始时间 {ts}) 数据！将使用0.0填补。")
                        feature_list.append(0.0)

                # (广播和添加到 self.data 的逻辑不变)
                feature_array = np.array(feature_list, dtype=np.float32)
                feature_broadcastable = feature_array.reshape(-1, 1, 1, 1)
                feature_full_spatial = np.broadcast_to(feature_broadcastable, (len(self.time_coords), 1, H, W))
                self.data = np.concatenate([self.data, feature_full_spatial], axis=1)
                self.feature_names.append(feature_name)  # 添加到特征名列表
                logger.info(f"成功添加 IMF 特征 '{feature_name}' 作为全局特征。")
        # --- ✨✨✨ 修改结束 ---

        logger.info(f"数据加载和处理完成 (目标月份: {self.target_month})。总特征数: {self.data.shape[1]}")
        logger.info("=" * 20 + " 特征名称及其索引 " + "=" * 20)
        for i, name in enumerate(self.feature_names): logger.info(f"索引 {i}: {name}")
        logger.info("=" * 58)

    def _prepare_data(self):
        self._load_data_from_files()
        atmos_feature_keywords = ['q', 'r', 't', 'u', 'v', 'w', 'z', 'tp']

        self.atmos_indices = []
        self.global_indices = []

        for i, name in enumerate(self.feature_names):
            is_atmos = False
            # 检查是否是大气变量 (例如 'q_850', 'z_500', 'tp')
            for keyword in atmos_feature_keywords:
                if name.startswith(keyword):
                    is_atmos = True
                    break

            if is_atmos:
                self.atmos_indices.append(i)
            else:
                # 其他所有特征 (dem, sst_anomaly_map, nino34, pmm_imf7 等) 都是全局特征 (g)
                self.global_indices.append(i)

        logger.info(f"特征已分离: {len(self.atmos_indices)} 个大气场特征, {len(self.global_indices)} 个全局特征。")
        try:
            self.atmos_feature_names = [self.feature_names[i] for i in self.atmos_indices]
            self.global_feature_names = [self.feature_names[i] for i in self.global_indices]

            # (可选, 但推荐) 打印出来以供调试
            logger.info(f"Atmos (X) 特征: {self.atmos_feature_names}")
            logger.info(f"Global (G) 特征: {self.global_feature_names}")

        except Exception as e:
            logger.error(f"构建特征子列表时出错: {e}")
            # --- ✨✨✨ [新增结束] ---

        if len(self.global_indices) == 0:
            logger.warning("未找到全局特征 (g)！MultiScaleGatedAttn 可能无法按预期工作。")

        self.data = np.nan_to_num(self.data, nan=0.0)

        # --- ✨✨✨ 修改: 使统计文件名包含目标月份 ---
        stats_dir = os.path.dirname(self.data_path)  # 获取父目录 (e.g., '../dataset')
        stats_filename = f'norm_stats_month{self.target_month}_sel_imfs.npz'
        stats_path = os.path.join(stats_dir, stats_filename)

        if self.type == 'train':
            # 计算包含当前月份特定IMFs的均值和标准差
            self.mean = np.mean(self.data, axis=(0, 2, 3), keepdims=True)
            self.std = np.std(self.data, axis=(0, 2, 3), keepdims=True)
            self.std[self.std < 1e-6] = 1.0
            np.savez(stats_path, mean=self.mean, std=self.std)
            logger.info(f"已计算并保存月份 {self.target_month} 的归一化统计数据到: {stats_path}")
        else:
            # 加载特定月份的统计文件
            if not os.path.exists(stats_path):
                raise FileNotFoundError(
                    f"验证/测试所需的统计文件未找到: {stats_path}。请确保已为月份 {self.target_month} 运行过训练集。")
            stats = np.load(stats_path)
            self.mean = stats['mean']
            self.std = stats['std']
            # ✨ 校验加载的统计数据维度是否匹配当前数据
            if self.mean.shape[1] != self.data.shape[1] or self.std.shape[1] != self.data.shape[1]:
                logger.error(
                    f"加载的统计数据通道数 ({self.mean.shape[1]}) 与当前数据通道数 ({self.data.shape[1]}) 不匹配！文件: {stats_path}")
                raise ValueError("统计数据维度不匹配，请检查是否加载了正确的月份统计文件或重新生成统计文件。")
            logger.info(f"已加载月份 {self.target_month} 的归一化统计数据从: {stats_path}")
        # --- ✨✨✨ 修改结束 ---

        # ... (构建样本序列的逻辑保持不变) ...
        logger.info("构建样本序列...")
        # (省略了样本构建代码以保持简洁，假设它按原样工作)
        time_dim = pd.to_datetime(self.time_coords);
        mask = np.isin(time_dim.month, self.required_months)
        filtered_time = time_dim[mask];
        filtered_data = self.data[mask]
        all_samples = [];
        start_month = self.target_month - self.input_len
        unique_years = sorted(list(set(pd.to_datetime(filtered_time).year)))
        for year in unique_years:
            year_mask = (pd.to_datetime(filtered_time).year == year);
            year_data = filtered_data[year_mask];
            year_time = filtered_time[year_mask]
            for t in range(len(year_data) - self.total_seq_len + 1):
                if year_time[t].month == start_month: sequence = year_data[
                                                                 t: t + self.total_seq_len]; all_samples.append(
                    sequence); break
        self.processed_samples = all_samples;
        logger.info(f"为 '{self.type}' 集 (月份 {self.target_month}) 构建了 {len(self.processed_samples)} 个样本。")

    def __getitem__(self, index):
        full_sequence = self.processed_samples[index]
        x_sequence_full = full_sequence[:self.input_len]
        y_sequence_full = full_sequence[self.input_len:]

        # (x_resized 逻辑保持不变)
        x_resized = np.zeros((self.input_len, x_sequence_full.shape[1], self.target_h, self.target_w), dtype=np.float32)
        for t in range(self.input_len):
            for c in range(x_sequence_full.shape[1]):
                x_resized[t, c, :, :] = cv2.resize(x_sequence_full[t, c, :, :], (self.target_w, self.target_h),
                                                   interpolation=cv2.INTER_CUBIC)

        # (y_sequence_cropped 逻辑保持不变)
        if self.lat_slice and self.lon_slice:
            y_sequence_cropped = y_sequence_full[:, :, self.lat_slice, self.lon_slice]
        else:
            y_sequence_cropped = y_sequence_full

        # (y_resized 逻辑保持不变)
        y_resized = np.zeros((self.output_len, y_sequence_cropped.shape[1], self.target_h, self.target_w),
                             dtype=np.float32)
        for t in range(self.output_len):
            for c in range(y_sequence_cropped.shape[1]):
                y_resized[t, c, :, :] = cv2.resize(y_sequence_cropped[t, c, :, :], (self.target_w, self.target_h),
                                                   interpolation=cv2.INTER_CUBIC)

        y_resized_tp_only = y_resized[:, 0:1, :, :]  # 只取 'tp' 作为Y

        # ✨✨✨ 自动归一化: 这里的 self.mean 和 self.std 已经包含了IMFs的均值和标准差
        # 'x_norm' 将包含所有输入特征，包括新的IMFs，并被正确归一化
        x_norm = (x_resized - self.mean) / (self.std + 1e-8)

        # 2. 归一化目标 (y_norm)
        y_norm = (y_resized_tp_only - self.mean[:, 0:1, :, :]) / (self.std[:, 0:1, :, :] + 1e-8)
        DataY = torch.from_numpy(y_norm).float()

        # --- ✨✨✨ [修改] 分离特征为 X 和 G ✨✨✨ ---

        # 3. 提取大气场特征 (x)
        # x_norm 形状 (T_in, C_all, H, W)
        x_norm_atmos = x_norm[:, self.atmos_indices, :, :]
        DataX_SpatioTemporal = torch.from_numpy(x_norm_atmos).float()

        # 4. 提取全局/背景特征 (g)
        x_norm_global = x_norm[:, self.global_indices, :, :]
        DataX_Global = torch.from_numpy(x_norm_global).float()

        # --- ✨✨✨ [修改结束] ✨✨✨ ---

        # 返回三个张量
        return DataX_SpatioTemporal, DataX_Global, DataY

    def __len__(self):
        if self.shuffle_flag and self.type == 'train':
            np.random.shuffle(self.processed_samples)
        return len(self.processed_samples)