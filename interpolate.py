from datetime import timedelta
import numpy as np
from scipy.interpolate import interp1d

def interpolate_gps(gps_data, target_datetime, range_minutes=30):
    """
    对GPS轨迹点进行三次样条插值

    Args:
        target_datetime: 目标时间
        range_minutes: 搜索范围（分钟）

    Returns:
        (latitude, longitude) 或 None
    """
    if gps_data is None:
        return None

    print(f"尝试在 {range_minutes} 分钟范围内进行三次样条插值")

    # 找到目标时间前后的轨迹点
    time_range = timedelta(minutes=range_minutes)
    start_time = target_datetime - time_range
    end_time = target_datetime + time_range

    # 筛选时间范围内的数据
    mask = (gps_data["datetime"] >= start_time) & (gps_data["datetime"] <= end_time)
    nearby_data = gps_data[mask].copy()

    print(f"时间范围 {start_time} 到 {end_time}")
    print(f"找到 {len(nearby_data)} 个GPS点在时间范围内")

    if len(nearby_data) < 4:  # 三次样条至少需要4个点
        print(
            f"时间范围内GPS点不足（{len(nearby_data)}个），需要至少4个点进行三次样条插值"
        )
        # 如果点数不足，降级为线性插值
        return linear_interpolation(gps_data, target_datetime, nearby_data)

    # 按时间排序
    nearby_data = nearby_data.sort_values("datetime")

    # 打印附近的GPS点用于调试
    print("附近的GPS点:")
    for idx, row in nearby_data.head(10).iterrows():
        print(f"  {row['datetime']} - {row['latitude']:.6f}, {row['longitude']:.6f}")

    try:
        # 将时间转换为数值（秒为单位）
        base_time = nearby_data["datetime"].iloc[0]
        time_values = np.array(
            [(dt - base_time).total_seconds() for dt in nearby_data["datetime"]]
        )

        # 获取经纬度
        lats = nearby_data["latitude"].values
        lons = nearby_data["longitude"].values

        # 计算目标时间对应的数值
        target_time_value = (target_datetime - base_time).total_seconds()

        # 检查目标时间是否在插值范围内
        if target_time_value < time_values[0] or target_time_value > time_values[-1]:
            print("目标时间超出GPS数据范围，使用最近点")
            if target_time_value < time_values[0]:
                return float(lats[0]), float(lons[0])
            else:
                return float(lats[-1]), float(lons[-1])

        # 创建三次样条插值函数
        lat_interp = interp1d(
            time_values,
            lats,
            kind="cubic",
            bounds_error=True,
        )
        lon_interp = interp1d(
            time_values,
            lons,
            kind="cubic",
            bounds_error=True,
        )

        # 计算插值结果
        interpolated_lat = float(lat_interp(target_time_value))
        interpolated_lon = float(lon_interp(target_time_value))

        print(f"三次样条插值结果: ({interpolated_lat:.6f}, {interpolated_lon:.6f})")

        return interpolated_lat, interpolated_lon

    except Exception as e:
        print(f"三次样条插值失败: {e}，降级为线性插值")
        return linear_interpolation(gps_data, target_datetime, nearby_data)


def linear_interpolation(gps_data, target_datetime, nearby_data):
    """
    线性插值备用方案

    Args:
        target_datetime: 目标时间
        nearby_data: 附近的GPS数据

    Returns:
        (latitude, longitude) 或 None
    """
    if len(nearby_data) < 2:
        if len(nearby_data) == 1:
            row = nearby_data.iloc[0]
            return float(row["latitude"]), float(row["longitude"])
        return None

    # 分别找到目标时间前后的点
    before_points = nearby_data[nearby_data["datetime"] <= target_datetime]
    after_points = nearby_data[nearby_data["datetime"] > target_datetime]

    print(f"目标时间前的点数: {len(before_points)}")
    print(f"目标时间后的点数: {len(after_points)}")

    # 如果目标时间前后都有点，进行线性插值
    if len(before_points) > 0 and len(after_points) > 0:
        # 获取最接近的前后两点
        point_before = before_points.iloc[-1]
        point_after = after_points.iloc[0]

        print(f"线性插值使用点:")
        print(
            f"  前点: {point_before['datetime']} ({point_before['latitude']:.6f}, {point_before['longitude']:.6f})"
        )
        print(
            f"  后点: {point_after['datetime']} ({point_after['latitude']:.6f}, {point_after['longitude']:.6f})"
        )

        # 线性插值
        total_time_diff = (
            point_after["datetime"] - point_before["datetime"]
        ).total_seconds()
        target_time_diff = (target_datetime - point_before["datetime"]).total_seconds()

        if total_time_diff == 0:
            return float(point_before["latitude"]), float(point_before["longitude"])

        # 计算插值比例
        ratio = target_time_diff / total_time_diff

        # 线性插值坐标
        interpolated_lat = point_before["latitude"] + ratio * (
            point_after["latitude"] - point_before["latitude"]
        )
        interpolated_lon = point_before["longitude"] + ratio * (
            point_after["longitude"] - point_before["longitude"]
        )

        print(f"线性插值结果: ({interpolated_lat:.6f}, {interpolated_lon:.6f})")
        print(f"插值比例: {ratio:.3f}")

        return float(interpolated_lat), float(interpolated_lon)

    # 如果只有一边有点，使用最接近的点
    elif len(before_points) > 0:
        print("只找到目标时间前的点，使用最接近的")
        closest_point = before_points.iloc[-1]
        return float(closest_point["latitude"]), float(closest_point["longitude"])

    elif len(after_points) > 0:
        print("只找到目标时间后的点，使用最接近的")
        closest_point = after_points.iloc[0]
        return float(closest_point["latitude"]), float(closest_point["longitude"])

    else:
        print("无法找到目标时间前后的GPS点")
        return None
