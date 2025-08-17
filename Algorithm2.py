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
    ky = 110_574.0                                 # m/deg (lat)
    return lon_deg * kx, lat_deg * ky, {"lat0": lat0, "kx": kx, "ky": ky}

def m_to_lonlat(x_m: np.ndarray, y_m: np.ndarray, meta: dict):
    return x_m / meta["kx"], y_m / meta["ky"]

# -------------------------------
# 读数据：tdrive_data/*.txt|*.csv
# 期望列：user_id / timestamp / lon / lat（或无表头四列）
# ------------------------------
TS_FMT = "%Y-%m-%d %H:%M:%S"
_ID_ALIASES  = {"user_id","uid","anon_id","id","taxi_id","vehicle_id","driver_id"}
_TS_ALIASES  = {"timestamp","time","datetime","date_time","dateTime"}
_LON_ALIASES = {"lon","lng","longitude","x"}
_LAT_ALIASES = {"lat","latitude","y"}

def _has_header(first_line: str) -> bool:
    toks = [t.strip().lower() for t in first_line.strip().split(",")]
    return any(t in (_ID_ALIASES|_TS_ALIASES|_LON_ALIASES|_LAT_ALIASES) for t in toks)

def _rename_cols(cols):
    out=[]
    for c in cols:
        cl=str(c).strip().lower()
        if   cl in _ID_ALIASES:  out.append("user_id")
        elif cl in _TS_ALIASES:  out.append("timestamp")
        elif cl in _LON_ALIASES: out.append("lon")
        elif cl in _LAT_ALIASES: out.append("lat")
        else: out.append(cl)
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
    files = sorted(glob.glob(os.path.join(folder,"*.txt")) +
                   glob.glob(os.path.join(folder,"*.csv")))
    if not files:
        raise FileNotFoundError(f"No txt/csv found in: {folder}")

    dfs=[]
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

        df = df.dropna(subset=["user_id","timestamp","lon","lat"])
        df["user_id"] = df["user_id"].astype(np.int64)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["user_id","timestamp","lon","lat"])
    out = out.sort_values(["user_id","timestamp"]).reset_index(drop=True)
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

    ix_lo = max(0, ix0 - rx); ix_hi = min(nx - 1, ix0 + rx)
    iy_lo = max(0, iy0 - rx); iy_hi = min(ny - 1, iy0 + rx)

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
        pub  = df_pub_xy_m[[rid_col, "x_m", "y_m"]].set_index(rid_col).loc[true.index]
        x_t, y_t = true["x_m"].to_numpy(), true["y_m"].to_numpy()
        x_z, y_z = pub["x_m"].to_numpy(),  pub["y_m"].to_numpy()
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
# Geo-I 平面拉普拉斯噪声
# r ~ Gamma(k=2, theta=1/epsilon), theta ~ U[0,2π)
# -------------------------------
def planar_laplace_noise(epsilon: float, n: int):
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    if epsilon == 0 or n == 0:
        return np.zeros(n), np.zeros(n)
    r = np.random.gamma(shape=2.0, scale=1.0/epsilon, size=n)
    ang = np.random.uniform(0.0, 2.0*np.pi, size=n)
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


# -------------------------------
# 算法2：全局置换 + 本地扰动
# -------------------------------
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
      out_df：列 [anon_id, timestamp, lon, lat, orig_lon, orig_lat, user_id, anon_map]
    """
    np.random.seed(seed)

    # --- 1) 全局置换 ---
    users = df["user_id"].drop_duplicates().to_numpy()
    perm  = users.copy()
    np.random.shuffle(perm)  # 论文里默认随机置换；不强制去除固定点
    mapping = {u_new: int(u_old) for u_new, u_old in zip(perm, users)}
    # 注意：上式是“原用户 -> 匿名ID”的映射；我们令匿名ID为 1..N 的紧致序号更利于导出
    # 为保持可重复且紧致，这里改为：原用户排序后，打乱为 [1..N]
    users_sorted = np.sort(users)
    anon_ids = np.arange(1, len(users_sorted)+1, dtype=int)
    np.random.shuffle(anon_ids)
    map_orig_to_anon = {int(u): int(a) for u, a in zip(users_sorted, anon_ids)}

    out = df.copy()
    out["anon_id"] = out["user_id"].map(map_orig_to_anon).astype(int)

    # --- 2) 本地扰动（在米制下）---
    # 经纬 -> 米
    x_m, y_m, meta = lonlat_to_m(out["lon"].to_numpy(float), out["lat"].to_numpy(float))

    # 噪声
    dx, dy = planar_laplace_noise(epsilon, len(out))
    x2, y2 = x_m + dx, y_m + dy

    # 可选边界裁剪（避免落在极远处；按原数据范围 ±1% 外扩）
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

    # 可记录映射表（导出或审计）
    map_df = pd.DataFrame({"user_id": users_sorted, "anon_id": anon_ids}).sort_values("anon_id")

    # ... 在 algo2_global_shuffle_and_pl 的 return 前 ...

    # 将内部计算好的米制坐标附加到输出DataFrame中
    out["orig_x_m"] = x_m
    out["orig_y_m"] = y_m
    out["pub_x_m"] = x2
    out["pub_y_m"] = y2

    return out, map_df, meta

# -------------------------------
# 指标：location-privacy 与 utility
# -------------------------------
def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def compute_metrics(out_df: pd.DataFrame, epsilon: float, window: str = "5min"):
    # AOD/ADE: 所有点的位移距离（米）平均
    disp = haversine_m(out_df["orig_lon"], out_df["orig_lat"], out_df["lon"], out_df["lat"])
    aod = float(disp.mean())
    med_disp = float(disp.median())

    # FDE: 每个用户的“末点位移”的平均
    last = out_df.sort_values("timestamp").groupby("anon_id").tail(1)
    fde = float(haversine_m(last["orig_lon"], last["orig_lat"], last["lon"], last["lat"]).mean())

    # k/BRP（用于横向参照）：按时间窗口统计并发用户数
    tmp = out_df.copy()
    tmp["bin"] = pd.to_datetime(tmp["timestamp"]).dt.floor(window)
    k_by_bin = tmp.groupby("bin")["anon_id"].nunique()
    k_mean = float(k_by_bin.mean()) if len(k_by_bin) else 0.0
    brp = float((1.0 / k_by_bin).mean()) if len(k_by_bin) else 0.0

    # 额外给出理论期望：E[r]=2/epsilon（平面拉普拉斯半径期望，作参考）
    theo_E_r = float(2.0/epsilon) if epsilon > 0 else float("inf")

    return {
        "epsilon": float(epsilon),
        "AOD_m": aod,              # 同点级 ADE
        "median_disp_m": med_disp,
        "FDE_m": fde,
        "k_anonymity_mean": k_mean,
        "BRP": brp,
        "theory_E_r_m": theo_E_r
    }

# -------------------------------
# CLI
# -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Algorithm 2 (global shuffle + Geo-I PL noise)")
    ap.add_argument("--data_dir", type=str, default="tdrive_data", help="数据目录（含若干 .txt/.csv）")
    ap.add_argument("--out_csv", type=str, default=None, help="输出文件（默认写入 data_dir/algo2_output.csv）")
    ap.add_argument("--map_csv", type=str, default=None, help="可选：输出 user->anon 映射表")
    ap.add_argument("--metrics_csv", type=str, default=None, help="可选：输出指标到CSV")
    ap.add_argument("--epsilon", type=float, required=True, help="Geo-I 平面拉普拉斯噪声参数 ε (>=0)")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（全局置换与噪声）")
    ap.add_argument("--window", type=str, default="5min", help="用于 k/BRP 统计的时间窗口")
    ap.add_argument("--no_bbox", action="store_true", help="不做边界裁剪")
    args = ap.parse_args()

    df = load_tdrive_folder(args.data_dir)
    #df = df.head(100).copy()
    print("=== 数据诊断 ===")
    print(f"总行数: {len(df)}")
    print(f"用户数: {df['user_id'].nunique()}")
    print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")

    out_df, map_df, meta = algo2_global_shuffle_and_pl(
        df, epsilon=args.epsilon, seed=args.seed, keep_bbox=(not args.no_bbox)
    )

    # # 评估指标
    # metrics = compute_metrics(out_df, epsilon=args.epsilon, window=args.window)
    # print("=== 指标（Algorithm 2）===")
    # for k, v in metrics.items():
    #     print(f"{k}: {v}")
    #
    # # ====== LP 评估（贝叶斯最优重映射） ======
    # # 网格大小建议与目标 SQL 同量级/2；若你已算出 theory_E_r_m，可取 cell≈max(50m, theory_E_r_m/2)
    # theory_E_r_m = metrics["theory_E_r_m"]
    # cell_m = max(50.0, theory_E_r_m / 2.0)
    # prior, meta = build_prior_grid(df_true_xy["x_m"], df_true_xy["y_m"], cell_m=cell_m, smooth=1.0)
    #
    # # 搜索半径取 4~5 倍理论噪声，保证覆盖足够候选
    # radius_m = max(500.0, 5.0 * theory_E_r_m)
    #
    # lp_mean_m, lp_median_m = compute_LP_mean_median(
    #     df_true_xy, df_pub_xy, epsilon, prior, meta, radius_m,
    #     rid_col=("rid" if "rid" in df_true_xy.columns and "rid" in df_pub_xy.columns else None)
    # )
    #
    # metrics["LP_mean_m"] = lp_mean_m
    # metrics["LP_median_m"] = lp_median_m
    # print(f"LP_mean_m: {lp_mean_m:.3f}")
    # print(f"LP_median_m: {lp_median_m:.3f}")
    # ... main() 函数的前半部分保持不变 ...

    # ======================================================
    # 1. 评估您的主算法 (Algo 2: Shuffle + PL)
    # ======================================================
    print("\n\n=== 评估主算法 (Algo 2: Shuffle + PL) ===")
    out_df, map_df, lonlat_meta = algo2_global_shuffle_and_pl(
        df, epsilon=args.epsilon, seed=args.seed, keep_bbox=(not args.no_bbox)
    )

    # 评估指标
    metrics = compute_metrics(out_df, epsilon=args.epsilon, window=args.window)
    print("--- 基础指标 ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # --- LP 评估 ---
    # 准备真实坐标和发布坐标（米制）
    # --- LP 评估 ---
    # 直接使用算法函数返回的、更可靠的米制坐标
    df_true_xy = out_df[["orig_x_m", "orig_y_m"]].rename(columns={"orig_x_m": "x_m", "orig_y_m": "y_m"})
    df_pub_xy = out_df[["pub_x_m", "pub_y_m"]].rename(columns={"pub_x_m": "x_m", "pub_y_m": "y_m"})

    theory_E_r_m = metrics["theory_E_r_m"]
    cell_m = max(50.0, theory_E_r_m / 2.0)
    # 使用修改后的 df_true_xy 来构建先验
    prior, prior_meta = build_prior_grid(df_true_xy["x_m"], df_true_xy["y_m"], cell_m=cell_m, smooth=1.0)
    radius_m = max(500.0, 5.0 * theory_E_r_m)

    lp_mean_m, lp_median_m = compute_LP_mean_median(
        df_true_xy, df_pub_xy, args.epsilon, prior, prior_meta, radius_m
    )
    metrics["LP_mean_m"] = lp_mean_m
    metrics["LP_median_m"] = lp_median_m
    print("--- 攻击者指标 ---")
    print(f"LP_mean_m: {lp_mean_m:.3f}")
    print(f"LP_median_m: {lp_median_m:.3f}")

    # ======================================================
    # 2. 评估对照组算法 (Algo 1: PL only)
    # ======================================================
    print("\n\n=== 评估对照组 (Algo 1: PL Only) ===")
    baseline_out_df, _, _ = algo1_local_pl_only(
        df, epsilon=args.epsilon, seed=args.seed, keep_bbox=(not args.no_bbox)
    )

    # 评估指标
    baseline_metrics = compute_metrics(baseline_out_df, epsilon=args.epsilon, window=args.window)
    print("--- 基础指标 ---")
    for k, v in baseline_metrics.items():
        print(f"{k}: {v}")

    # --- LP 评估 (使用相同的 prior) ---
    # --- LP 评估 (使用相同的 prior) ---
    # 直接使用算法函数返回的、更可靠的米制坐标
    baseline_true_xy = baseline_out_df[["orig_x_m", "orig_y_m"]].rename(columns={"orig_x_m": "x_m", "orig_y_m": "y_m"})
    baseline_pub_xy = baseline_out_df[["pub_x_m", "pub_y_m"]].rename(columns={"pub_x_m": "x_m", "pub_y_m": "y_m"})

    baseline_lp_mean_m, baseline_lp_median_m = compute_LP_mean_median(
        baseline_true_xy, baseline_pub_xy, args.epsilon, prior, prior_meta, radius_m
    )
    baseline_metrics["LP_mean_m"] = baseline_lp_mean_m
    baseline_metrics["LP_median_m"] = baseline_lp_median_m
    print("--- 攻击者指标 ---")
    print(f"LP_mean_m: {baseline_lp_mean_m:.3f}")
    print(f"LP_median_m: {baseline_lp_median_m:.3f}")

    # # 导出
    # out_path = args.out_csv or os.path.join(args.data_dir, "algo2_output.csv")
    # out_df[["anon_id","timestamp","lon","lat","orig_lon","orig_lat"]].to_csv(out_path, index=False)
    # print(f"输出: {out_path}")
    #
    # if args.map_csv:
    #     map_df.to_csv(args.map_csv, index=False)
    #     print(f"映射表: {args.map_csv}")
    # if args.metrics_csv:
    #     pd.DataFrame([metrics]).to_csv(args.metrics_csv, index=False)
    #     print(f"指标已保存: {args.metrics_csv}")

if __name__ == "__main__":
    main()
