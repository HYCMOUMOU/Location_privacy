# -*- coding: utf-8 -*-
"""
Algorithm2.py — 轨迹置换匿名（窗口级候选 + 匈牙利/贪心 + 最大置换距离）

用法:
  python Algorithm2.py --data_dir tdrive_data --out algo2_fast_output.csv
  # 只读前 N 个文件 / 前 M 行:
  python Algorithm2.py --data_dir tdrive_data --limit_files 10 --limit_rows 50000

依赖: numpy, pandas
可选: scikit-learn (KDTree), scipy (linear_sum_assignment)
"""

import os, re, glob, math, argparse
from collections import Counter
import numpy as np
import pandas as pd

# -------------------------------
# 可选依赖
# -------------------------------
try:
    from sklearn.neighbors import KDTree
    SK_KDTREE = True
except Exception:
    KDTree = None
    SK_KDTREE = False

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_OK = True
except Exception:
    linear_sum_assignment = None
    SCIPY_OK = False


# -------------------------------
# 算法参数（建议从这组开始，再逐步收紧）
# -------------------------------
RANDOM_SEED = 2025

# 小实验：限制读取数量（None 表示不限制）
LIMIT_FILES = 10
LIMIT_ROWS  = 100000

# 时间窗口（分钟）
TIME_WINDOW_MIN = 45

# 候选空间半径（米）
RS = 2500.0     # 起点半径
RE = 5000.0     # 终点半径（无就设 None 或 0）

# 运动学阈值
DV      = 25.0      # 速度差阈值 (km/h)
DTHETA  = 90.0     # 朝向差阈值 (deg)
LEN_TOL = 1000.0    # 轨迹段长度差阈值 (m)

# 代价函数权重（空间已做半径归一化）
W_S = 1.00   # 起点距离/RS
W_E = 0.70   # 终点距离/RE
W_V = 0.10   # |Δv|/DV
W_T = 0.05   # angdiff/ DTHETA
W_L = 0.0005 # |Δlen|/LEN_TOL

# 匹配规模阈值（<= 用匈牙利，否则贪心）
MAX_HUNGARIAN_N = 60

# BRP 先验网格（米）
CELL_SIZE_M  = 50.0         # 备用
PRIOR_GRID_M = 2000.0       # 稳定性更好（1500~3000 均可）

# 自身匹配代价（禁止对角时用）
IDENTITY_PENALTY = 1e12

# ★ 最大置换距离（米）：起点&终点都须 ≤ 此阈值才允许置换；否则自匹配
MAX_SWAP_DIST_M = 3000.0


# -------------------------------
# 工具函数
# -------------------------------
# -------------------------------
# 工具函数（替换这两个）
# -------------------------------
def is_lonlat_like(a):
    """
    既支持 pandas Series 也支持 numpy ndarray。
    判断数值是否像经纬度范围：[-180, 180]。
    """
    a = np.asarray(a, dtype=float)           # 统一成 ndarray
    if a.size == 0:
        return False
    mask = ~np.isnan(a)
    if not np.any(mask):                     # 全是 NaN
        return False
    mn = float(np.nanmin(a))
    mx = float(np.nanmax(a))
    return (mn >= -180.0 - 1e-6) and (mx <= 180.0 + 1e-6)

def to_xy_meters(df, xcol, ycol):
    """
    如果像经纬度，则做近似墨卡托到米的投影；否则按原始数值使用。
    """
    x_series = pd.to_numeric(df[xcol], errors='coerce')
    y_series = pd.to_numeric(df[ycol], errors='coerce')
    x = x_series.to_numpy(dtype=float)
    y = y_series.to_numpy(dtype=float)

    if is_lonlat_like(x) and is_lonlat_like(y):
        lon, lat = x, y
        lat0 = np.nanmean(lat) if not np.isnan(lat).all() else 0.0
        R = 6371000.0
        x_m = np.deg2rad(lon - np.nanmean(lon)) * R * math.cos(math.radians(lat0))
        y_m = np.deg2rad(lat - np.nanmean(lat)) * R
        return x_m, y_m
    else:
        return x, y


def robust_parse_time(s):
    try:
        v = float(s)
        if v > 1e12: v /= 1000.0
        if v > 1e10: v /= 1000.0
        return pd.to_datetime(int(v), unit='s', utc=False)
    except Exception:
        return pd.to_datetime(s, utc=False, errors='coerce')

def load_folder_tdrive(data_dir, limit_files=None, limit_rows=None):
    paths = sorted([p for p in glob.glob(os.path.join(data_dir, "*")) if os.path.isfile(p)])
    if limit_files is not None:
        paths = paths[:int(limit_files)]

    rows, total = [], 0
    for p in paths:
        uid_from_name = re.findall(r"(\d+)", os.path.basename(p))
        uid_from_name = int(uid_from_name[0]) if uid_from_name else None
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if limit_rows is not None and total >= int(limit_rows): break
                line = line.strip()
                if not line: continue
                parts = [c.strip() for c in (line.split(",") if "," in line else line.split())]
                parts = [c for c in parts if c != ""]
                if len(parts) < 3: continue

                if len(parts) >= 4:
                    try:
                        uid = int(float(parts[0]))
                        ts  = robust_parse_time(parts[1])
                        xraw, yraw = parts[2], parts[3]
                    except Exception:
                        uid = uid_from_name if uid_from_name is not None else -1
                        ts  = robust_parse_time(parts[0])
                        xraw, yraw = parts[1], parts[2]
                else:
                    uid = uid_from_name if uid_from_name is not None else -1
                    ts  = robust_parse_time(parts[0])
                    xraw, yraw = parts[1], parts[2]

                if pd.isna(ts): continue
                rows.append((uid, ts, xraw, yraw)); total += 1

    if not rows:
        raise RuntimeError(f"在 {data_dir} 没读到有效数据")

    df = pd.DataFrame(rows, columns=["user_id","t","xraw","yraw"])
    x_m, y_m = to_xy_meters(df, "xraw", "yraw")
    df["x"], df["y"] = x_m, y_m
    df = df.drop(columns=["xraw","yraw"]).sort_values(["user_id","t"]).reset_index(drop=True)
    return df

def print_diagnostics(df):
    n = len(df)
    n_users = df["user_id"].nunique()
    u_counts = df["user_id"].value_counts().to_dict()
    tmin, tmax = df["t"].min(), df["t"].max()
    hrs = (tmax - tmin).total_seconds() / 3600.0
    print("=== 数据诊断 ===")
    print(f"总行数: {n}")
    print(f"用户数: {n_users}")
    print(f"用户分布: {u_counts}")
    print(f"时间范围: {tmin} 到 {tmax}")
    print(f"时间跨度: {hrs:.1f} 小时")

def window_iter(df_xy, win_minutes=60):
    df = df_xy.copy()
    w = (df["t"].view("int64") // 10**9) // int(win_minutes*60)
    df["w"] = w.astype(np.int64)
    for (win, uid), g in df.groupby(["w", "user_id"], sort=True):
        yield int(win), int(uid), g.sort_values("t")

def segment_stats(xy, tt):
    if len(xy) < 2: return 0.0, 0.0, 0.0
    diffs = xy[1:] - xy[:-1]
    seglen = np.sqrt((diffs**2).sum(axis=1))
    length = float(seglen.sum())
    dt = float(tt[-1] - tt[0])
    vbar = 0.0 if dt <= 0 else (length / dt) * 3.6  # m/s -> km/h
    dx, dy = (xy[-1] - xy[0]).tolist()
    thetabar = math.degrees(math.atan2(dy, dx)) % 360.0
    return vbar, thetabar, length

def angdiff(a, b):
    return abs((a - b + 180) % 360 - 180)

def angdiff_vec(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)

def resample_to_len(xy_src, target_len):
    xy_src = np.asarray(xy_src, dtype=float)
    n = int(xy_src.shape[0]) if xy_src.ndim == 2 else 0
    target_len = int(target_len)
    if target_len <= 1:
        return xy_src[[0], :] if n >= 1 else np.zeros((1,2), dtype=float)
    if n == 0:  return np.zeros((target_len,2), dtype=float)
    if n == 1:  return np.repeat(xy_src, target_len, axis=0)
    if n == target_len: return xy_src.copy()
    xp_src = np.linspace(0.0, 1.0, num=n)
    xp_tgt = np.linspace(0.0, 1.0, num=target_len)
    x_new = np.interp(xp_tgt, xp_src, xy_src[:,0])
    y_new = np.interp(xp_tgt, xp_src, xy_src[:,1])
    return np.column_stack([x_new, y_new])

def hungarian_np(cost):
    n = cost.shape[0]
    rows = list(range(n))
    cols = set(range(n))
    r_idx, c_idx = [], []
    for i in rows:
        cand_cols = list(cols)
        j = cand_cols[int(np.argmin(cost[i, cand_cols]))]
        r_idx.append(i); c_idx.append(j)
        cols.remove(j)
    return np.array(r_idx), np.array(c_idx)


# -------------------------------
# 核心算法
# -------------------------------
def algo2_shuffle_fast(traj_df,
                       win_minutes=TIME_WINDOW_MIN,
                       cell_size=None,
                       rs=RS, re=RE, dv=DV, dtheta=DTHETA,
                       len_tol=LEN_TOL,
                       max_hungarian_n=MAX_HUNGARIAN_N,
                       identity_penalty=IDENTITY_PENALTY,
                       max_swap_dist=MAX_SWAP_DIST_M):
    """
    窗口级候选 + 三段兜底 + 无近邻允许自匹配 + 最大置换距离硬约束
    返回:
        out_df: [user_id, t, x, y]
        metrics: {'BRP','ADE','FDE'}
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # 1) 生成窗口内“轨迹段”（每用户每窗一个）
    seg_list = []
    for w, u, g in window_iter(traj_df, win_minutes=win_minutes):
        xy = g[["x","y"]].to_numpy(dtype=np.float32)
        tt = (g["t"].astype("int64") // 10**9).to_numpy()
        if len(xy) < 2: continue
        vbar, thetabar, length = segment_stats(xy, tt)
        seg_list.append(dict(
            w=w, u=u, xy=xy, t=tt,
            stats=(vbar, thetabar, length),
            start=xy[0], end=xy[-1]
        ))
    seg_df = pd.DataFrame(seg_list)
    if seg_df.empty:
        return traj_df.copy(), {"BRP":1.0, "ADE":0.0, "FDE":0.0}

    perm_assign = {}            # (w, gi) -> gj
    brp_candidates = {}         # gi -> 候选(全局索引列表)
    total_swaps, total_self = 0, 0

    # 2) 按窗口处理
    for w, seg_w in seg_df.groupby("w", sort=False):
        idxs = seg_w.index.to_numpy()
        n = len(idxs)
        if n < 2:
            gi = idxs[0]
            perm_assign[(w, gi)] = gi
            brp_candidates[gi]   = [gi]
            continue

        starts = np.vstack(seg_df.loc[idxs, "start"].to_numpy())
        ends   = np.vstack(seg_df.loc[idxs, "end"].to_numpy())
        stats  = np.vstack(seg_df.loc[idxs, "stats"].to_numpy())  # [v, theta, len]
        v_arr, th_arr, len_arr = stats[:,0], stats[:,1], stats[:,2]

        # KDTree
        if SK_KDTREE:
            tree_s = KDTree(starts)
            tree_e = KDTree(ends) if (re is not None and re > 0) else None
        else:
            tree_s = tree_e = None
            ds_all = np.sqrt(((starts[:,None,:]-starts[None,:,:])**2).sum(axis=2))
            de_all = np.sqrt(((ends[:,None,:]-ends[None,:,:])**2).sum(axis=2)) if (re is not None and re > 0) else None

        BIG = identity_penalty
        use_hungarian = (n <= max_hungarian_n)

        # ---------- 候选构造（带最大置换距离） ----------
        def build_candidates(ii: int):
            """
            优先在 {rs,re} 半径内找候选；若无 -> 逐步扩半径；
            再施加“最大置换距离”硬约束（起点与终点都 <= max_swap_dist）。
            若仍无候选 -> 返回空，让外层允许自匹配。
            """
            muls = (1.0, 1.5, 2.0, 3.0)
            RS_MAX = 8000.0
            RE_MAX = 20000.0 if (re is not None and re > 0) else RS_MAX

            for m in muls:
                rad_s = min(rs * m, RS_MAX)
                rad_e = min((re if (re is not None and re > 0) else rs) * m, RE_MAX)

                if SK_KDTREE:
                    idx_s = tree_s.query_radius(starts[[ii]], rad_s)[0]
                    if tree_e is not None:
                        idx_e = tree_e.query_radius(ends[[ii]], rad_e)[0]
                    else:
                        idx_e = np.arange(n)
                else:
                    idx_s = np.where(ds_all[ii] <= rad_s)[0]
                    idx_e = np.where(de_all[ii] <= rad_e)[0] if de_all is not None else np.arange(n)

                base = np.intersect1d(idx_s, idx_e)
                base = base[base != ii]
                if base.size == 0:
                    base = idx_s[idx_s != ii]
                if base.size == 0:
                    continue  # 扩半径继续

                # 最大置换距离硬约束（原始米）
                ds_base = np.linalg.norm(starts[base] - starts[ii], axis=1)
                mask = (ds_base <= max_swap_dist)
                if re is not None and re > 0:
                    de_base = np.linalg.norm(ends[base] - ends[ii], axis=1)
                    mask &= (de_base <= max_swap_dist)
                base = base[mask]
                if base.size == 0:
                    continue

                # 运动学过滤；若过滤空，则退回 base
                dvv = np.abs(v_arr[ii] - v_arr[base]) <= dv
                dtt = angdiff_vec(th_arr[ii], th_arr[base]) <= dtheta
                dll = np.abs(len_arr[ii] - len_arr[base]) <= len_tol
                cand = base[dvv & dtt & dll]
                return cand if cand.size else base

            # 无任何空间近邻 -> 交给外层允许自匹配
            return np.array([], dtype=int)

        # 归一化常数
        RS_N = max(rs, 1.0)
        RE_N = max(re if (re is not None and re > 0) else rs, 1.0)
        DV_N = max(dv, 1.0)
        DT_N = max(dtheta, 1.0)
        LL_N = max(len_tol, 1.0)

        if use_hungarian:
            cost = np.full((n, n), BIG, dtype=np.float64)
            for ii in range(n):
                cand = build_candidates(ii)
                if cand.size == 0:
                    # 无近邻 -> 允许自匹配（对角为0）
                    cost[ii, ii] = 0.0
                    brp_candidates[idxs[ii]] = [idxs[ii]]
                    continue

                # 有候选 -> 禁止自匹配
                cost[ii, ii] = BIG

                # 空间项（归一化）+ 运动学项
                ds = np.linalg.norm(starts[ii] - starts[cand], axis=1) / RS_N
                de = np.linalg.norm(ends[ii]   - ends[cand],   axis=1) / RE_N
                dvv = np.abs(v_arr[ii]  - v_arr[cand]) / DV_N
                dtt = angdiff_vec(th_arr[ii], th_arr[cand]) / DT_N
                dll = np.abs(len_arr[ii]- len_arr[cand])   / LL_N

                c = W_S*ds + W_E*de + W_V*dvv + W_T*dtt + W_L*dll
                cost[ii, cand] = 1e-6 + c
                brp_candidates[idxs[ii]] = [idxs[j] for j in cand]

            if SCIPY_OK:
                r_idx, c_idx = linear_sum_assignment(cost)
            else:
                r_idx, c_idx = hungarian_np(cost)

            for r, c in zip(r_idx, c_idx):
                gi = idxs[r]; gj = idxs[c]

                # 复核原始距离，超阈值则自匹配
                ds_raw = float(np.linalg.norm(starts[r] - starts[c]))
                if re is not None and re > 0:
                    de_raw = float(np.linalg.norm(ends[r] - ends[c]))
                    too_far = (ds_raw > max_swap_dist) or (de_raw > max_swap_dist)
                else:
                    too_far = (ds_raw > max_swap_dist)

                if too_far:
                    gj = gi
                    total_self += 1
                else:
                    if gj != gi: total_swaps += 1
                    else:        total_self  += 1

                perm_assign[(w, gi)] = gj

        else:
            # 贪心
            assigned = set()
            cand_map = {}
            for ii in range(n):
                cand = build_candidates(ii)
                cand_map[ii] = cand
                brp_candidates[idxs[ii]] = [idxs[j] for j in cand] if cand.size else [idxs[ii]]

            order = sorted(range(n), key=lambda i: len(cand_map[i]) if len(cand_map[i])>0 else 9e9)
            for ii in order:
                gi = idxs[ii]
                cand = [j for j in cand_map[ii] if j not in assigned]
                if not cand:
                    gj = gi
                else:
                    cand = np.array(cand, dtype=int)
                    ds = np.linalg.norm(starts[ii] - starts[cand], axis=1) / RS_N
                    de = np.linalg.norm(ends[ii]   - ends[cand],   axis=1) / RE_N
                    dvv = np.abs(v_arr[ii]  - v_arr[cand]) / DV_N
                    dtt = angdiff_vec(th_arr[ii], th_arr[cand]) / DT_N
                    dll = np.abs(len_arr[ii]- len_arr[cand])   / LL_N
                    cc  = W_S*ds + W_E*de + W_V*dvv + W_T*dtt + W_L*dll
                    jj  = cand[int(np.argmin(cc))]

                    # 复核硬距离
                    ds_raw = float(np.linalg.norm(starts[ii] - starts[jj]))
                    if re is not None and re > 0:
                        de_raw = float(np.linalg.norm(ends[ii] - ends[jj]))
                        too_far = (ds_raw > max_swap_dist) or (de_raw > max_swap_dist)
                    else:
                        too_far = (ds_raw > max_swap_dist)

                    if too_far:
                        gj = gi
                    else:
                        assigned.add(jj)
                        gj = idxs[jj]

                perm_assign[(w, gi)] = gj
                if gj != gi: total_swaps += 1
                else:        total_self  += 1

        # 每窗诊断
        cand_sizes = [len(brp_candidates[idxs[i]]) for i in range(n)]
        has_cand   = np.mean(np.array(cand_sizes) > 0)
        multi_cand = np.mean(np.array(cand_sizes) > 1)
        print(f"[窗口 {w}] n={n}, 有候选率={has_cand:.1%}, 候选>1率={multi_cand:.1%}")

    # 3) 应用置换
    seg_map = {gi: perm_assign.get((seg_df.at[gi,"w"], gi), gi) for gi in seg_df.index}
    out_rows = []
    for gi, row in seg_df.iterrows():
        gj = seg_map.get(gi, gi)
        xy_new = seg_df.at[gj, "xy"]
        xy_old = row["xy"]
        if xy_new.shape[0] < 2:
            xy_new = np.vstack([xy_new, xy_new])
        xy_new_interp = resample_to_len(xy_new, len(xy_old))
        for (xv, yv), tv in zip(xy_new_interp, row["t"]):
            out_rows.append((row["w"], row["u"], pd.to_datetime(tv, unit="s"), xv, yv))
    out_df = pd.DataFrame(out_rows, columns=["w","user_id","t","x","y"]).sort_values(["user_id","t"]).reset_index(drop=True)

    # 4) 指标：BRP / ADE / FDE
    def grid_key(pt, g=None):
        g0 = PRIOR_GRID_M if (PRIOR_GRID_M and PRIOR_GRID_M > 0) else (cell_size if cell_size else CELL_SIZE_M)
        return (int(math.floor(pt[0] / g0)), int(math.floor(pt[1] / g0)))

    starts_orig = [tuple(s) for s in np.vstack(seg_df["start"].to_numpy())]
    prior_counts = Counter([grid_key(s) for s in starts_orig])
    total_segs = len(starts_orig)
    prior_prob  = {k: v/total_segs for k, v in prior_counts.items()}

    BRP_list, ADE_list, FDE_list = [], [], []
    for gi, row in seg_df.iterrows():
        C = brp_candidates.get(gi, [gi])
        denom = sum(prior_prob.get(grid_key(seg_df.at[c, "start"]), 1e-9) for c in C)
        num   = prior_prob.get(grid_key(row["start"]), 1e-9)
        BRP_list.append(0.0 if denom == 0 else num/denom)

        gj = seg_map.get(gi, gi)
        xy_old = row["xy"]
        xy_new = resample_to_len(seg_df.at[gj,"xy"], len(xy_old))
        diff   = xy_old - xy_new
        ade = float(np.mean(np.sqrt(np.sum(diff**2, axis=1))))
        fde = float(np.linalg.norm(xy_old[-1] - xy_new[-1]))
        ADE_list.append(ade); FDE_list.append(fde)

    metrics = {"BRP": float(np.mean(BRP_list)),
               "ADE": float(np.mean(ADE_list)),
               "FDE": float(np.mean(FDE_list))}

    # 5) 统计
    all_pairs = len(seg_df)
    print("=== 置换统计 ===")
    print(f"总置换数: {total_swaps}")
    print(f"自匹配数: {total_self}")
    print(f"置换率: {total_swaps/max(1,all_pairs):.1%}")

    return out_df[["user_id","t","x","y"]], metrics


# -------------------------------
# CLI
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="tdrive_data", help="数据文件夹(含若干txt/csv)")
    parser.add_argument("--out", type=str, default="algo2_fast_output.csv", help="输出CSV文件名")
    parser.add_argument("--limit_files", type=int, default=LIMIT_FILES, help="最多读取多少个文件")
    parser.add_argument("--limit_rows",  type=int, default=LIMIT_ROWS,  help="最多读取多少行")
    parser.add_argument("--cell_size",   type=float, default=CELL_SIZE_M, help="备用网格大小(米)")
    parser.add_argument("--max_swap_dist", type=float, default=MAX_SWAP_DIST_M, help="最大置换距离(米)")
    args = parser.parse_args()

    df_xy = load_folder_tdrive(args.data_dir, args.limit_files, args.limit_rows)
    print_diagnostics(df_xy)
    print(f"Loaded rows={len(df_xy)} users={df_xy['user_id'].nunique()}")

    out_df, metrics = algo2_shuffle_fast(
        df_xy,
        win_minutes=TIME_WINDOW_MIN,
        cell_size=args.cell_size,
        rs=RS, re=RE, dv=DV, dtheta=DTHETA,
        len_tol=LEN_TOL,
        max_hungarian_n=MAX_HUNGARIAN_N,
        identity_penalty=IDENTITY_PENALTY,
        max_swap_dist=args.max_swap_dist
    )

    print("=== 指标 ===")
    print(metrics)
    out_path = os.path.join(args.data_dir, args.out) if os.path.isdir(args.data_dir) else args.out
    out_df.to_csv(out_path, index=False)
    print(f"输出: {out_path}")

if __name__ == "__main__":
    main()
