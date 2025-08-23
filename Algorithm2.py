# Filename: Algorithm.py
# -*- coding: utf-8 -*-

import os, glob, argparse, math
import numpy as np
import pandas as pd


# -------------------------------
# 经纬度 <-> 米制（局部等距近似）
# -------------------------------
def lonlat_to_m(lon_deg: np.ndarray, lat_deg: np.ndarray):
    lat0 = float(np.median(lat_deg))
    kx = 111_320.0 * math.cos(math.radians(lat0))  # m/deg (lon)
    ky = 110_574.0  # m/deg (lat)
    return lon_deg * kx, lat_deg * ky, {"lat0": lat0, "kx": kx, "ky": ky}


def m_to_lonlat(x_m: np.ndarray, y_m: np.ndarray, meta: dict):
    return x_m / meta["kx"], y_m / meta["ky"]


# -------------------------------
# 读数据：tdrive_data/*.txt|*.csv
# 期望列：user_id / timestamp / lon / lat（或无表头四列）
# ------------------------------
TS_FMT = "%Y-%m-%d %H:%M:%S"
_ID_ALIASES = {"user_id", "uid", "anon_id", "id", "taxi_id", "vehicle_id", "driver_id"}
_TS_ALIASES = {"timestamp", "time", "datetime", "date_time", "dateTime"}
_LON_ALIASES = {"lon", "lng", "longitude", "x"}
_LAT_ALIASES = {"lat", "latitude", "y"}


def _has_header(first_line: str) -> bool:
    toks = [t.strip().lower() for t in first_line.strip().split(",")]
    return any(t in (_ID_ALIASES | _TS_ALIASES | _LON_ALIASES | _LAT_ALIASES) for t in toks)


def _rename_cols(cols):
    out = []
    for c in cols:
        cl = str(c).strip().lower()
        if cl in _ID_ALIASES:
            out.append("user_id")
        elif cl in _TS_ALIASES:
            out.append("timestamp")
        elif cl in _LON_ALIASES:
            out.append("lon")
        elif cl in _LAT_ALIASES:
            out.append("lat")
        else:
            out.append(cl)
    return out


def _parse_ts_fast(col: pd.Series) -> pd.Series:
    s = col.astype("string").str.strip()
    # 关键修改：直接初始化为带时区的dtype，以匹配后续所有解析操作
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")

    p1 = pd.to_datetime(s, format=TS_FMT, errors="coerce", utc=True)
    out.loc[:] = p1
    left = out.isna()
    if left.any():
        s_left = s[left]
        mask_num = s_left.str.fullmatch(r"\d+").fillna(False)
        if mask_num.any():
            num = pd.to_numeric(s_left[mask_num], errors="coerce")
            unit = "ms" if num.median(skipna=True) and num.median() > 1e12 else "s"
            out.loc[s_left.index[mask_num]] = pd.to_datetime(num, unit=unit, errors="coerce", utc=True)
    left = out.isna()
    if left.any():
        out.loc[left] = pd.to_datetime(s[left], errors="coerce", utc=True)

    return out.dt.tz_localize(None)


def load_tdrive_folder(folder: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "*.txt")) +
                   glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No txt/csv found in: {folder}")

    dfs = []
    for f in files:
        df = pd.read_csv(
            f, sep=",", header=None, engine="c",
            names=["user_id", "timestamp", "lon", "lat"],
            usecols=[0, 1, 2, 3],
            encoding="utf-8-sig"
        )
        # -------------------- 修改结束 --------------------

        # 只保留核心四列（若文件有多余列会被丢弃）
        keep = [c for c in ["user_id", "timestamp", "lon", "lat"] if c in df.columns]
        df = df[keep]

        # user_id 读完再转数值，避免 "anon_id" 触发异常
        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")

        # 时间戳快速解析（固定格式 -> epoch -> 兜底）
        df["timestamp"] = _parse_ts_fast(df["timestamp"])

        # 类型清洗
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")

        df = df.dropna(subset=["user_id", "timestamp", "lon", "lat"])
        df["user_id"] = df["user_id"].astype(np.int64)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["user_id", "timestamp", "lon", "lat"])
    out = out.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return out


def build_prior_grid(x_m, y_m, cell_m=200.0, smooth=1.0):
    """
    基于原始坐标(米)构建空间先验：把点落到 cell_m 网格，计数后做+smooth 平滑。
    返回 prior 网格以及坐标映射参数。
    """
    x_m = np.asarray(x_m, dtype=float)
    y_m = np.asarray(y_m, dtype=float)
    xmin, xmax = float(x_m.min()), float(x_m.max())
    ymin, ymax = float(y_m.min()), float(y_m.max())
    nx = int(np.ceil((xmax - xmin) / cell_m)) + 1
    ny = int(np.ceil((ymax - ymin) / cell_m)) + 1
    counts = np.zeros((ny, nx), dtype=np.float64)

    ix = np.clip(((x_m - xmin) / cell_m).astype(int), 0, nx - 1)
    iy = np.clip(((y_m - ymin) / cell_m).astype(int), 0, ny - 1)
    np.add.at(counts, (iy, ix), 1.0)

    prior = (counts + smooth)
    prior /= prior.sum()  # 归一化成概率

    meta = {"xmin": xmin, "ymin": ymin, "cell": cell_m, "ny": ny, "nx": nx}
    return prior, meta


def _cell_center(ix, iy, meta):
    cx = meta["xmin"] + (ix + 0.5) * meta["cell"]
    cy = meta["ymin"] + (iy + 0.5) * meta["cell"]
    return cx, cy


def bayes_est_point(zx, zy, eps, prior, meta, radius_m):
    """
    给定发布点 z=(zx,zy)、先验 prior、epsilon 和搜索半径，返回攻击者的 MAP 估计 (x_hat,y_hat)。
    似然按 planar Laplace：L ∝ exp(-eps * d)，常数因子对 argmax 不重要。
    """
    cell = meta["cell"]
    ny, nx = prior.shape
    rx = int(np.ceil(radius_m / cell))
    ix0 = int(np.round((zx - meta["xmin"]) / cell))
    iy0 = int(np.round((zy - meta["ymin"]) / cell))

    ix_lo = max(0, ix0 - rx);
    ix_hi = min(nx - 1, ix0 + rx)
    iy_lo = max(0, iy0 - rx);
    iy_hi = min(ny - 1, iy0 + rx)

    best_s, best_xy = -1.0, (zx, zy)
    # 穷举邻域网格（半径通常取 3~5 倍理论噪声）
    for iy in range(iy_lo, iy_hi + 1):
        cy = meta["ymin"] + (iy + 0.5) * cell
        for ix in range(ix_lo, ix_hi + 1):
            cx = meta["xmin"] + (ix + 0.5) * cell
            d = float(np.hypot(zx - cx, zy - cy))
            s = prior[iy, ix] * np.exp(-eps * d)
            if s > best_s:
                best_s = s
                best_xy = (cx, cy)
    return best_xy


def compute_LP_mean_median(df_true_xy_m, df_pub_xy_m, eps, prior, meta,
                           radius_m, rid_col=None):
    """
    计算 LP：对每个发布点做贝叶斯重映射并与其真实点求距离。
    - df_true_xy_m: 必须含 x_m, y_m（原始）
    - df_pub_xy_m : 必须含 x_m, y_m（发布/扰动后）
    - 若二者有行号/主键（如 'rid'），可传 rid_col=该列名进行对齐；否则按索引一一对应
    返回: lp_mean, lp_median
    """
    if rid_col and (rid_col in df_true_xy_m.columns) and (rid_col in df_pub_xy_m.columns):
        true = df_true_xy_m[[rid_col, "x_m", "y_m"]].set_index(rid_col)
        pub = df_pub_xy_m[[rid_col, "x_m", "y_m"]].set_index(rid_col).loc[true.index]
        x_t, y_t = true["x_m"].to_numpy(), true["y_m"].to_numpy()
        x_z, y_z = pub["x_m"].to_numpy(), pub["y_m"].to_numpy()
    else:
        # 按当前顺序对齐
        x_t = df_true_xy_m["x_m"].to_numpy()
        y_t = df_true_xy_m["y_m"].to_numpy()
        x_z = df_pub_xy_m["x_m"].to_numpy()
        y_z = df_pub_xy_m["y_m"].to_numpy()
        n = min(len(x_t), len(x_z))
        x_t, y_t, x_z, y_z = x_t[:n], y_t[:n], x_z[:n], y_z[:n]

    dists = np.empty_like(x_t, dtype=np.float64)
    for i in range(len(x_t)):
        xh, yh = bayes_est_point(x_z[i], y_z[i], eps, prior, meta, radius_m)
        dists[i] = float(np.hypot(x_t[i] - xh, y_t[i] - yh))

    return float(np.mean(dists)), float(np.median(dists))


# -------------------------------
# 轨迹级攻击函数（新增）
# -------------------------------
def attack_with_known_id(result_df, epsilon, prior, prior_meta, radius_m):
    """
    算法1的情况：攻击者知道user_id对应关系
    """
    all_errors = []

    # 直接在包含了所有坐标的 result_df 上操作
    for uid in result_df['user_id'].unique():
        traj = result_df[result_df['user_id'] == uid].sort_values('timestamp')

        for i in range(len(traj)):
            point = traj.iloc[i]

            # 贝叶斯估计
            x_hat, y_hat = bayes_est_point(
                point['pub_x_m'], point['pub_y_m'],
                epsilon, prior, prior_meta, radius_m
            )

            # 计算误差
            error = np.hypot(point['orig_x_m'] - x_hat,
                             point['orig_y_m'] - y_hat)
            all_errors.append(error)

    if not all_errors:
        return 0.0, 0.0

    return np.mean(all_errors), np.median(all_errors)


def attack_without_known_id(result_df, epsilon, prior, prior_meta, radius_m):
    """
    算法2的情况：攻击者需要先破解shuffle映射（修正版：使用单一数据源）
    """
    from scipy.optimize import linear_sum_assignment

    # 从单一的结果DataFrame中获取用户列表
    true_users = np.sort(result_df['user_id'].unique())
    pub_users = np.sort(result_df['anon_id'].unique())
    n_users = min(len(true_users), len(pub_users))

    if n_users == 0:
        return 0.0, 0.0

    print(f"  构建{n_users}x{n_users}轨迹距离矩阵...")
    dist_matrix = np.full((n_users, n_users), np.inf)

    # 为每个用户（真实和匿名）预先计算特征，避免重复计算
    true_user_features = {uid: extract_trajectory_features(result_df[result_df['user_id'] == uid]) for uid in
                          true_users}
    pub_user_features = {aid: extract_trajectory_features(result_df[result_df['anon_id'] == aid]) for aid in pub_users}

    for i, true_uid in enumerate(true_users):
        for j, pub_uid in enumerate(pub_users):
            features_true = true_user_features[true_uid]
            features_pub = pub_user_features[pub_uid]
            dist = np.linalg.norm(features_true - features_pub)
            dist_matrix[i, j] = dist

    # 添加噪声到距离矩阵，模拟不确定性
    noise_scale = 100 / epsilon if epsilon > 0 else 1e5
    noise = np.random.normal(0, noise_scale, dist_matrix.shape)
    dist_matrix_noisy = dist_matrix + np.abs(noise)

    # 替换所有无效值 (inf or NaN)
    invalid_mask = ~np.isfinite(dist_matrix_noisy)
    if np.any(invalid_mask):
        finite_vals = dist_matrix_noisy[~invalid_mask]
        if finite_vals.size > 0:
            replacement_val = np.max(finite_vals) * 10 + 1
            dist_matrix_noisy[invalid_mask] = replacement_val
        else:
            dist_matrix_noisy[invalid_mask] = 1e9

    print("  运行匈牙利算法寻找最优用户匹配...")
    row_ind, col_ind = linear_sum_assignment(dist_matrix_noisy)

    # 评估匹配准确性
    correct_matches = 0
    true_to_anon_map = result_df.drop_duplicates('user_id').set_index('user_id')['anon_id']

    all_errors = []
    for r, c in zip(row_ind, col_ind):
        true_uid = true_users[r]
        matched_pub_uid = pub_users[c]

        # 检查匹配是否正确
        if true_uid in true_to_anon_map and true_to_anon_map[true_uid] == matched_pub_uid:
            correct_matches += 1

        # 获取真实轨迹和被匹配上的匿名轨迹
        true_points = result_df[result_df['user_id'] == true_uid].sort_values('timestamp')
        pub_points = result_df[result_df['anon_id'] == matched_pub_uid].sort_values('timestamp')

        # 使用merge确保时间戳对齐，避免错位
        merged_traj = pd.merge(
            true_points[['timestamp', 'orig_x_m', 'orig_y_m']],
            pub_points[['timestamp', 'pub_x_m', 'pub_y_m']],
            on='timestamp'
        )

        if merged_traj.empty:
            continue

        for _, row in merged_traj.iterrows():
            # 贝叶斯估计
            x_hat, y_hat = bayes_est_point(
                row['pub_x_m'], row['pub_y_m'],
                epsilon, prior, prior_meta, radius_m
            )
            # 计算误差
            # 此处可以安全地访问 'orig_x_m' 因为它来自 result_df
            error = np.hypot(row['orig_x_m'] - x_hat, row['orig_y_m'] - y_hat)
            all_errors.append(error)

    accuracy = (correct_matches / n_users * 100) if n_users > 0 else 0
    print(f"  用户重识别准确率: {correct_matches}/{n_users} = {accuracy:.1f}%")

    if accuracy < 50:
        print(f"  ✓ Shuffle有效！准确率仅{accuracy:.1f}%")
    else:
        print(f"  ⚠ Shuffle效果有限，准确率达{accuracy:.1f}%")

    if not all_errors:
        return 0.0, 0.0

    return np.mean(all_errors), np.median(all_errors)

def extract_trajectory_features(traj):
    """
    提取轨迹的统计特征 (修改后，可处理NaN)
    """
    if len(traj) == 0:
        return np.zeros(10)

    features = []

    # 空间特征
    features.append(traj['lon'].mean())  # 质心经度
    features.append(traj['lat'].mean())  # 质心纬度

    # 对标准差计算结果进行处理，将单点轨迹产生的NaN替换为0
    features.append(traj['lon'].std(ddof=0))
    features.append(traj['lat'].std(ddof=0))

    # 时间特征
    if 'timestamp' in traj.columns:
        timestamps = pd.to_datetime(traj['timestamp'])
        features.append(len(traj))  # 轨迹点数

        # 时间跨度（小时）
        time_span = (timestamps.max() - timestamps.min()).total_seconds() / 3600
        features.append(time_span)

        # 平均速度（如果有连续点）
        if len(traj) > 1:
            distances = []
            times = []
            for i in range(len(traj) - 1):
                p1 = traj.iloc[i]
                p2 = traj.iloc[i + 1]
                # 注意: haversine_m 函数需要确保存在
                # 如果haversine_m未在文件顶部定义，您需要添加它
                dist = haversine_m(p1['lon'], p1['lat'], p2['lon'], p2['lat'])
                distances.append(dist)

                time_diff = (timestamps.iloc[i + 1] - timestamps.iloc[i]).total_seconds()
                if time_diff > 0:
                    times.append(time_diff)

            if times:
                avg_speed = np.mean([d / t for d, t in zip(distances, times)])
                features.append(avg_speed)
            else:
                features.append(0)
        else:
            features.append(0)

    # 填充到固定长度
    while len(features) < 10:
        features.append(0)

    # 【核心改动】在返回前，使用np.nan_to_num确保没有无效值
    # 这会将所有 NaN 替换为 0.0，同时也会处理 inf (无穷大)
    features_array = np.nan_to_num(np.array(features[:10]), nan=0.0, posinf=0.0, neginf=0.0)

    return features_array

# -------------------------------
# Geo-I 平面拉普拉斯噪声
# r ~ Gamma(k=2, theta=1/epsilon), theta ~ U[0,2π)
# -------------------------------
def planar_laplace_noise(epsilon: float, n: int):
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    if epsilon == 0 or n == 0:
        return np.zeros(n), np.zeros(n)
    r = np.random.gamma(shape=2.0, scale=1.0 / epsilon, size=n)
    ang = np.random.uniform(0.0, 2.0 * np.pi, size=n)
    return r * np.cos(ang), r * np.sin(ang)


# -------------------------------
# 算法1：仅本地扰动（对照组/基线）
# -------------------------------
def algo1_local_pl_only(df: pd.DataFrame,
                        epsilon: float,
                        seed: int = 0,
                        keep_bbox: bool = True):
    """
    对照组：只添加 PL_ε 噪声，不进行全局置换。
    """
    np.random.seed(seed)
    out = df.copy()

    # --- 本地扰动（在米制下）---
    # 经纬 -> 米
    x_m, y_m, meta = lonlat_to_m(out["lon"].to_numpy(float), out["lat"].to_numpy(float))

    # 噪声
    dx, dy = planar_laplace_noise(epsilon, len(out))
    x2, y2 = x_m + dx, y_m + dy

    # 可选边界裁剪
    if keep_bbox:
        xmin, xmax = x_m.min(), x_m.max()
        ymin, ymax = y_m.min(), y_m.max()
        pad_x = 0.01 * max(1.0, xmax - xmin)
        pad_y = 0.01 * max(1.0, ymax - ymin)
        x2 = np.clip(x2, xmin - pad_x, xmax + pad_x)
        y2 = np.clip(y2, ymin - pad_y, ymax + pad_y)

    # 米 -> 经纬
    lon2, lat2 = m_to_lonlat(x2, y2, meta)

    out["orig_lon"] = out["lon"].to_numpy(float)
    out["orig_lat"] = out["lat"].to_numpy(float)
    out["lon"] = lon2
    out["lat"] = lat2

    # 对照组没有 anon_id，但为了后续代码结构统一，可保留 user_id
    out["anon_id"] = out["user_id"]

    # ... 在 algo1_local_pl_only 的 return 前 ...

    # 将内部计算好的米制坐标附加到输出DataFrame中
    out["orig_x_m"] = x_m
    out["orig_y_m"] = y_m
    out["pub_x_m"] = x2
    out["pub_y_m"] = y2

    return out, None, meta


def algo2_global_shuffle_and_pl(df: pd.DataFrame,
                                epsilon: float,
                                seed: int = 0,
                                keep_bbox: bool = True):
    """
    输入：原始经纬度数据（列：user_id,timestamp,lon,lat）
    流程：
      1) 全局一次性随机置换 user_id（整份数据映射一致）
      2) 经纬度 -> 米制；添加 PL_ε 噪声；可选边界裁剪；再转回经纬度
    返回：
      out_df：包含匿名化结果和米制坐标的DataFrame
    """
    np.random.seed(seed)
    out = df.copy()

    # --- 1) 全局置换 ---
    # 获取唯一的、排好序的原始 user_id
    users_sorted = np.sort(df["user_id"].unique())
    num_users = len(users_sorted)

    # 创建一套新的匿名ID并随机打乱
    anon_ids = np.arange(1, num_users + 1, dtype=int)
    np.random.shuffle(anon_ids)

    # 创建从原始ID到匿名ID的映射字典
    map_orig_to_anon = {int(u): int(a) for u, a in zip(users_sorted, anon_ids)}

    # 应用映射，生成 anon_id 列
    out["anon_id"] = out["user_id"].map(map_orig_to_anon)

    # 检查一下，确保映射后没有空值
    if out["anon_id"].isnull().any():
        print("警告：ID映射后出现空值，请检查 user_id 数据类型是否一致！")
        # 将无法映射的 anon_id 暂时用 user_id 填充，避免程序崩溃
        out["anon_id"].fillna(out["user_id"], inplace=True)

    out["anon_id"] = out["anon_id"].astype(int)

    # --- 2) 本地扰动（在米制下）---
    # 经纬 -> 米
    x_m, y_m, meta = lonlat_to_m(out["lon"].to_numpy(float), out["lat"].to_numpy(float))

    # 噪声
    dx, dy = planar_laplace_noise(epsilon, len(out))
    x2, y2 = x_m + dx, y_m + dy

    # 可选边界裁剪
    if keep_bbox:
        xmin, xmax = x_m.min(), x_m.max()
        ymin, ymax = y_m.min(), y_m.max()
        pad_x = 0.01 * max(1.0, xmax - xmin)
        pad_y = 0.01 * max(1.0, ymax - ymin)
        x2 = np.clip(x2, xmin - pad_x, xmax + pad_x)
        y2 = np.clip(y2, ymin - pad_y, ymax + pad_y)

    # 米 -> 经纬
    lon2, lat2 = m_to_lonlat(x2, y2, meta)

    out["orig_lon"] = out["lon"].to_numpy(float)
    out["orig_lat"] = out["lat"].to_numpy(float)
    out["lon"] = lon2
    out["lat"] = lat2

    # 将内部计算好的米制坐标附加到输出DataFrame中
    out["orig_x_m"] = x_m
    out["orig_y_m"] = y_m
    out["pub_x_m"] = x2
    out["pub_y_m"] = y2

    # 创建映射关系的DataFrame用于审计
    map_df = pd.DataFrame({"user_id": users_sorted, "anon_id": anon_ids}).sort_values("anon_id")

    return out, map_df, meta


# -------------------------------
# 指标：location-privacy 与 utility
# -------------------------------
def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def compute_metrics(out_df: pd.DataFrame, epsilon: float, window: str = "5min"):
    # AOD/ADE: 所有点的位移距离（米）平均
    disp = haversine_m(out_df["orig_lon"], out_df["orig_lat"], out_df["lon"], out_df["lat"])
    aod = float(disp.mean())
    med_disp = float(disp.median())

    # FDE: 每个用户的"末点位移"的平均
    last = out_df.sort_values("timestamp").groupby("anon_id").tail(1)
    fde = float(haversine_m(last["orig_lon"], last["orig_lat"], last["lon"], last["lat"]).mean())

    # k/BRP（用于横向参照）：按时间窗口统计并发用户数
    tmp = out_df.copy()
    tmp["bin"] = pd.to_datetime(tmp["timestamp"]).dt.floor(window)
    k_by_bin = tmp.groupby("bin")["anon_id"].nunique()
    k_mean = float(k_by_bin.mean()) if len(k_by_bin) else 0.0
    brp = float((1.0 / k_by_bin).mean()) if len(k_by_bin) else 0.0

    # 额外给出理论期望：E[r]=2/epsilon（平面拉普拉斯半径期望，作参考）
    theo_E_r = float(2.0 / epsilon) if epsilon > 0 else float("inf")

    return {
        "epsilon": float(epsilon),
        "AOD_m": aod,  # 同点级 ADE
        "median_disp_m": med_disp,
        "FDE_m": fde,
        "k_anonymity_mean": k_mean,
        "BRP": brp,
        "theory_E_r_m": theo_E_r
    }

def compute_range_query_error(
    df_true, df_pub, lonlat_meta, grid_size=100, return_counts=False
):
    """
    计算范围查询的误差，以评估数据效用（精度）。
    该函数模拟了“统计区域内点的总数”这一常见应用场景。

    Args:
        df_true (pd.DataFrame): 包含原始坐标的DataFrame ('orig_lon', 'orig_lat')
        df_pub (pd.DataFrame): 包含发布坐标的DataFrame ('lon', 'lat')
        lonlat_meta (dict): 经纬度与米制转换的元数据 (在此版本中未使用，但保留以便兼容)
        grid_size (int): 将地图划分的网格边长 (例如 100x100)
        return_counts (bool): 是否返回详细的计数矩阵和范围

    Returns:
        - return_counts=False (默认)：float(MAE)
        - return_counts=True ：(float(MAE), true_counts, pub_counts, extent)
          extent = [xmin, xmax, ymin, ymax]，可直接用于 imshow(..., extent=extent)
    """
    print(f"    正在计算 {grid_size}x{grid_size} 网格的范围查询误差...", end="", flush=True)

    # 0) 清洗并提取坐标
    t_lon = df_true["orig_lon"].to_numpy(dtype=float)
    t_lat = df_true["orig_lat"].to_numpy(dtype=float)
    p_lon = df_pub["lon"].to_numpy(dtype=float)
    p_lat = df_pub["lat"].to_numpy(dtype=float)

    # 1) 经纬 -> 米制
    true_x_m, true_y_m, _ = lonlat_to_m(t_lon, t_lat)
    pub_x_m,  pub_y_m,  _ = lonlat_to_m(p_lon, p_lat)

    # 2) 用“真值 + 发布”的合并范围建网格（避免发布点被夹到边界）
    xmin = min(true_x_m.min(), pub_x_m.min()); xmax = max(true_x_m.max(), pub_x_m.max())
    ymin = min(true_y_m.min(), pub_y_m.min()); ymax = max(true_y_m.max(), pub_y_m.max())
    dx = xmax - xmin; dy = ymax - ymin
    if dx == 0: dx = 1.0
    if dy == 0: dy = 1.0
    eps = 1e-6
    xmin -= eps * dx; xmax += eps * dx
    ymin -= eps * dy; ymax += eps * dy
    extent = [xmin, xmax, ymin, ymax]

    # 3) 计算真实计数
    true_counts = np.zeros((grid_size, grid_size), dtype=int)
    ix_true = np.clip(((true_x_m - xmin) / (xmax - xmin) * grid_size).astype(int), 0, grid_size - 1)
    iy_true = np.clip(((true_y_m - ymin) / (ymax - ymin) * grid_size).astype(int), 0, grid_size - 1)
    np.add.at(true_counts, (iy_true, ix_true), 1)

    # 4) 计算发布数据的计数
    pub_counts = np.zeros((grid_size, grid_size), dtype=int)
    ix_pub = np.clip(((pub_x_m - xmin) / (xmax - xmin) * grid_size).astype(int), 0, grid_size - 1)
    iy_pub = np.clip(((pub_y_m - ymin) / (ymax - ymin) * grid_size).astype(int), 0, grid_size - 1)
    np.add.at(pub_counts, (iy_pub, ix_pub), 1)

    # 5) 计算 MAE
    mae = float(np.mean(np.abs(true_counts - pub_counts)))
    print("完成.")

    if return_counts:
        return mae, true_counts, pub_counts, extent
    return mae


# -------------------------------
# CLI (最终版)
# -------------------------------

def main():
    # 1. 参数解析
    ap = argparse.ArgumentParser(description="Algorithm 2 (global shuffle + Geo-I PL noise)")
    ap.add_argument("--data_dir", type=str, default="tdrive_data", help="数据目录（含若干 .txt/.csv）")
    ap.add_argument("--epsilon", type=float, required=True, help="Geo-I 平面拉普拉斯噪声参数 ε (>=0)")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（全局置换与噪声）")
    ap.add_argument("--window", type=str, default="5min", help="用于 k/BRP 统计的时间窗口")
    ap.add_argument("--no_bbox", action="store_true", help="不做边界裁剪")
    args = ap.parse_args()

    # 2. 加载和清洗数据
    df = load_tdrive_folder(args.data_dir)

    beijing_bbox = {
        "lon_min": 115.7, "lon_max": 117.4,
        "lat_min": 39.4, "lat_max": 41.6
    }
    rows_before = len(df)
    df = df[
        (df['lon'].between(beijing_bbox["lon_min"], beijing_bbox["lon_max"])) &
        (df['lat'].between(beijing_bbox["lat_min"], beijing_bbox["lat_max"]))
        ].copy()
    rows_after = len(df)
    print(f"数据清洗：已根据北京地理范围过滤数据，从 {rows_before} 行减少到 {rows_after} 行。")

    # 取样本数据以加快测试速度
    df = df.head(10000).copy()  # 取10000行数据进行测试

    print("\n=== 数据诊断 ===")
    print(f"总行数: {len(df)}")
    print(f"用户数: {df['user_id'].nunique()}")
    print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")

    # ======================================================
    # 3. 评估主算法 (Algo 2: Shuffle + PL)
    # ======================================================
    print(f"\n\n=== 评估主算法 (Algo 2: Shuffle + PL) | ε = {args.epsilon} ===")
    out_df, _, lonlat_meta = algo2_global_shuffle_and_pl(
        df, epsilon=args.epsilon, seed=args.seed, keep_bbox=(not args.no_bbox)
    )

    metrics = compute_metrics(out_df, epsilon=args.epsilon, window=args.window)
    print("--- 基础指标 ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    mae = compute_range_query_error(out_df, out_df)
    metrics['MAE_RangeQuery'] = mae
    print(f"--- 数据效用指标 ---")
    print(f"范围查询MAE: {mae:.3f}")

    # # --- LP 评估 (使用简单的快照式攻击) ---
    # df_true_xy = out_df[["orig_x_m", "orig_y_m"]].rename(columns={"orig_x_m": "x_m", "orig_y_m": "y_m"})
    # df_pub_xy = out_df[["pub_x_m", "pub_y_m"]].rename(columns={"pub_x_m": "x_m", "pub_y_m": "y_m"})
    #
    # theory_E_r_m = metrics["theory_E_r_m"]
    # cell_m = max(50.0, theory_E_r_m / 2.0)
    # prior, prior_meta = build_prior_grid(df_true_xy["x_m"], df_true_xy["y_m"], cell_m=cell_m, smooth=1.0)
    # radius_m = max(500.0, 5.0 * theory_E_r_m)
    #
    # lp_mean_m, lp_median_m = compute_LP_mean_median(
    #     df_true_xy, df_pub_xy, args.epsilon, prior, prior_meta, radius_m
    # )
    # print("--- 攻击者指标 (快照攻击) ---")
    # print(f"LP_mean_m: {lp_mean_m:.3f}")
    # print(f"LP_median_m: {lp_median_m:.3f}")

    # ======================================================
    # 4. 评估对照组算法 (Algo 1: PL only)
    # ======================================================
    print(f"\n\n=== 评估对照组 (Algo 1: PL Only) | ε = {args.epsilon} ===")
    baseline_out_df, _, _ = algo1_local_pl_only(
        df, epsilon=args.epsilon, seed=args.seed, keep_bbox=(not args.no_bbox)
    )

    baseline_metrics = compute_metrics(baseline_out_df, epsilon=args.epsilon, window=args.window)
    print("--- 基础指标 ---")
    for k, v in baseline_metrics.items():
        print(f"{k}: {v}")

    # --- 新增：数据效用评估 (范围查询) ---
    baseline_mae = compute_range_query_error(baseline_out_df, baseline_out_df)
    baseline_metrics['MAE_RangeQuery'] = baseline_mae
    print(f"--- 数据效用指标 ---")
    print(f"范围查询MAE: {baseline_mae:.3f}")
    # # --- LP 评估 (使用相同的快照式攻击和相同的 prior) ---
    # baseline_true_xy = baseline_out_df[["orig_x_m", "orig_y_m"]].rename(columns={"orig_x_m": "x_m", "orig_y_m": "y_m"})
    # baseline_pub_xy = baseline_out_df[["pub_x_m", "pub_y_m"]].rename(columns={"pub_x_m": "x_m", "pub_y_m": "y_m"})
    #
    # baseline_lp_mean_m, baseline_lp_median_m = compute_LP_mean_median(
    #     baseline_true_xy, baseline_pub_xy, args.epsilon, prior, prior_meta, radius_m
    # )
    # print("--- 攻击者指标 (快照攻击) ---")
    # print(f"LP_mean_m: {baseline_lp_mean_m:.3f}")
    # print(f"LP_median_m: {baseline_lp_median_m:.3f}")

    # ======================================================
    # 5. 轨迹级攻击对比（这是体现Shuffle价值的关键）
    # ======================================================
    # print("\n\n=== 轨迹级攻击对比 ===")
    # print("轨迹攻击模拟用户重识别场景，攻击者试图链接整条轨迹")
    #
    # # 对算法1的轨迹攻击 (传入 baseline_out_df)
    # print("\n--- 算法1轨迹攻击 (攻击者知道user_id映射) ---")
    # algo1_traj_mean, algo1_traj_median = attack_with_known_id(
    #     baseline_out_df, args.epsilon, prior, prior_meta, radius_m
    # )
    # print(f"轨迹攻击 LP_mean_m: {algo1_traj_mean:.3f}")
    # print(f"轨迹攻击 LP_median_m: {algo1_traj_median:.3f}")
    #
    # # 对算法2的轨迹攻击
    # print("\n--- 算法2轨迹攻击 (攻击者需要破解shuffle) ---")
    # algo2_traj_mean, algo2_traj_median = attack_without_known_id(
    #     out_df, args.epsilon, prior, prior_meta, radius_m
    # )
    # print(f"轨迹攻击 LP_mean_m: {algo2_traj_mean:.3f}")
    # print(f"轨迹攻击 LP_median_m: {algo2_traj_median:.3f}")
    #
    # # ======================================================
    # # 6. 隐私保护效果总结
    # # ======================================================
    # print("\n\n=== 隐私保护效果总结 ===")
    #
    # # 快照攻击对比
    # snapshot_improvement = (
    #             (baseline_lp_mean_m - lp_mean_m) / baseline_lp_mean_m * 100) if baseline_lp_mean_m > 0 else 0
    # print(f"\n快照攻击（单点）：")
    # print(f"  算法1 LP: {baseline_lp_mean_m:.1f}m")
    # print(f"  算法2 LP: {lp_mean_m:.1f}m")
    # print(f"  改进: {snapshot_improvement:.1f}%")
    #
    # # 轨迹攻击对比
    # trajectory_improvement = ((algo2_traj_mean - algo1_traj_mean) / algo1_traj_mean * 100) if algo1_traj_mean > 0 else 0
    # print(f"\n轨迹攻击（整条轨迹）：")
    # print(f"  算法1 LP: {algo1_traj_mean:.1f}m")
    # print(f"  算法2 LP: {algo2_traj_mean:.1f}m")
    # print(f"  改进: {trajectory_improvement:.1f}%")
    #
    # # 隐私放大因子
    # n_users = df['user_id'].nunique()
    # amplification_factor = 1.0 / np.sqrt(n_users)
    # print(f"\n理论隐私放大因子:")
    # print(f"  用户数: {n_users}")
    # print(f"  放大因子: 1/√{n_users} = {amplification_factor:.4f}")
    #
    # # 核心结论
    # print("\n" + "=" * 50)
    # print("核心发现：")
    # if trajectory_improvement > 100:
    #     print(f"✓ Shuffle在轨迹级攻击下提供了 {trajectory_improvement:.0f}% 的隐私增强！")
    #     print(f"✓ 这证明了Shuffle模型在地理位置数据上的核心价值")
    # else:
    #     print(f"⚠ 轨迹级隐私增强为 {trajectory_improvement:.0f}%")
    #     print(f"⚠ 可能需要更多用户数据来体现Shuffle的效果")
    # print("=" * 50)


if __name__ == "__main__":
    main()
