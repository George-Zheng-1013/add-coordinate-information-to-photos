import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from PIL import Image
from PIL.ExifTags import TAGS
import piexif
import argparse
from scipy.interpolate import interp1d

import interpolate


class GeotagPhotos:
    def __init__(self, csv_file, photo_folder):
        """
        初始化地理标记照片类

        Args:
            csv_file: 包含UTC时间戳和GPS坐标的CSV文件路径
            photo_folder: 包含照片的文件夹路径
        """
        self.csv_file = csv_file
        self.photo_folder = photo_folder
        self.gps_data = None

    def load_gps_data(self):
        """加载GPS轨迹数据"""
        try:
            # 读取CSV文件
            self.gps_data = pd.read_csv(self.csv_file)

            print("CSV文件列名:", self.gps_data.columns.tolist())
            print("前3行数据:")
            print(self.gps_data.head(3))

            # 自动检测时间列名和坐标列名
            time_columns = ["dataTime", "timestamp", "datetime", "time", "Time"]
            lat_columns = ["latitude", "lat", "Latitude", "Lat"]
            lon_columns = ["longitude", "lon", "Longitude", "Lon"]

            time_col = None
            lat_col = None
            lon_col = None

            for col in time_columns:
                if col in self.gps_data.columns:
                    time_col = col
                    break

            for col in lat_columns:
                if col in self.gps_data.columns:
                    lat_col = col
                    break

            for col in lon_columns:
                if col in self.gps_data.columns:
                    lon_col = col
                    break

            if not all([time_col, lat_col, lon_col]):
                print("未找到必要的列，请确保CSV文件包含时间、纬度、经度列")
                print("可用列名:", self.gps_data.columns.tolist())
                return False

            print(f"使用列: 时间={time_col}, 纬度={lat_col}, 经度={lon_col}")

            # 统一列名
            if time_col != "dataTime":
                self.gps_data["dataTime"] = self.gps_data[time_col]
            if lat_col != "latitude":
                self.gps_data["latitude"] = self.gps_data[lat_col]
            if lon_col != "longitude":
                self.gps_data["longitude"] = self.gps_data[lon_col]

            # 将Unix时间戳转换为datetime对象
            sample_timestamp = self.gps_data["dataTime"].iloc[0]
            print(f"样本时间戳: {sample_timestamp}")

            # 如果时间戳大于1e10，则可能是毫秒，需要除以1000
            if sample_timestamp > 1e10:
                print("检测到毫秒级时间戳，转换为秒")
                self.gps_data["datetime"] = pd.to_datetime(
                    self.gps_data["dataTime"], unit="ms", utc=True
                )
            else:
                print("检测到秒级时间戳")
                self.gps_data["datetime"] = pd.to_datetime(
                    self.gps_data["dataTime"], unit="s", utc=True
                )

            print(f"转换后的datetime示例:")
            print(self.gps_data[["dataTime", "datetime"]].head(3))
            print(
                f"GPS数据时间范围: {self.gps_data['datetime'].min()} 到 {self.gps_data['datetime'].max()}"
            )
            print(f"成功加载 {len(self.gps_data)} 条GPS记录")
            return True

        except Exception as e:
            print(f"加载GPS数据失败: {e}")
            return False

    def get_photo_datetime(self, photo_path):
        """
        获取照片的拍摄时间（东八区时间）

        Args:
            photo_path: 照片文件路径

        Returns:
            datetime对象（UTC时间）或None
        """
        try:
            image = Image.open(photo_path)
            exif_dict = piexif.load(image.info.get("exif", b""))

            # 尝试获取拍摄时间
            datetime_original = None
            if piexif.ExifIFD.DateTimeOriginal in exif_dict["Exif"]:
                datetime_str = exif_dict["Exif"][
                    piexif.ExifIFD.DateTimeOriginal
                ].decode("utf-8")
                # 格式: "YYYY:MM:DD HH:MM:SS"
                datetime_original = datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")

                # 照片时间为东八区，转换为UTC
                china_tz = timezone(timedelta(hours=8))
                datetime_china = datetime_original.replace(tzinfo=china_tz)
                datetime_utc = datetime_china.astimezone(timezone.utc)

                return datetime_utc

        except Exception as e:
            print(f"无法读取照片 {photo_path} 的时间信息: {e}")

        return None

    def find_closest_gps(
        self, photo_datetime, max_time_diff_minutes=5, interpolation_range_minutes=30
    ):
        """
        查找最接近照片时间的GPS坐标，支持插值

        Args:
            photo_datetime: 照片拍摄时间（UTC）
            max_time_diff_minutes: 直接匹配的最大时间差（分钟）
            interpolation_range_minutes: 插值搜索范围（分钟）

        Returns:
            (latitude, longitude) 或 None
        """
        if self.gps_data is None or photo_datetime is None:
            return None

        print(f"查找照片时间 {photo_datetime} 对应的GPS坐标")

        # 计算时间差
        self.gps_data["time_diff"] = abs(self.gps_data["datetime"] - photo_datetime)

        # 找到最小时间差的记录
        min_idx = self.gps_data["time_diff"].idxmin()
        closest_record = self.gps_data.loc[min_idx]
        time_diff_minutes = closest_record["time_diff"].total_seconds() / 60

        print(f"最接近的GPS记录时间: {closest_record['datetime']}")
        print(f"时间差: {time_diff_minutes:.1f} 分钟")

        # 如果时间差在直接匹配范围内，直接返回
        if time_diff_minutes <= max_time_diff_minutes:
            print("使用直接匹配")
            return closest_record["latitude"], closest_record["longitude"]

        # 尝试插值
        print(f"时间差超过 {max_time_diff_minutes} 分钟，尝试插值...")
        interpolated_coords = interpolate.interpolate_gps(
            self.gps_data, photo_datetime, interpolation_range_minutes
        )

        if interpolated_coords is not None:
            print("使用插值结果")
            return interpolated_coords
        else:
            print(f"无法进行插值，时间差 {time_diff_minutes:.1f} 分钟超过阈值")
            return None

    def decimal_to_dms(self, decimal_coord):
        """
        将十进制坐标转换为度分秒格式

        Args:
            decimal_coord: 十进制坐标

        Returns:
            ((degrees, 1), (minutes, 1), (seconds_numerator, seconds_denominator))
        """
        is_positive = decimal_coord >= 0
        decimal_coord = abs(decimal_coord)

        degrees = int(decimal_coord)
        minutes = int((decimal_coord - degrees) * 60)
        seconds = ((decimal_coord - degrees) * 60 - minutes) * 60

        # 将秒转换为分数形式，保持精度
        seconds_numerator = int(seconds * 1000000)
        seconds_denominator = 1000000

        return ((degrees, 1), (minutes, 1), (seconds_numerator, seconds_denominator))

    def add_gps_to_photo(self, photo_path, latitude, longitude):
        """
        为照片添加GPS信息

        Args:
            photo_path: 照片文件路径
            latitude: 纬度
            longitude: 经度
        """
        try:
            # 读取现有EXIF数据
            image = Image.open(photo_path)
            exif_dict = piexif.load(image.info.get("exif", b""))

            # 转换坐标格式
            lat_dms = self.decimal_to_dms(latitude)
            lon_dms = self.decimal_to_dms(longitude)

            # 设置GPS信息
            gps_dict = {
                piexif.GPSIFD.GPSLatitude: lat_dms,
                piexif.GPSIFD.GPSLatitudeRef: "N" if latitude >= 0 else "S",
                piexif.GPSIFD.GPSLongitude: lon_dms,
                piexif.GPSIFD.GPSLongitudeRef: "E" if longitude >= 0 else "W",
            }

            exif_dict["GPS"] = gps_dict

            # 保存更新后的EXIF数据
            exif_bytes = piexif.dump(exif_dict)
            image.save(photo_path, exif=exif_bytes)

            print(
                f"已为 {os.path.basename(photo_path)} 添加GPS信息: {latitude:.6f}, {longitude:.6f}"
            )

        except Exception as e:
            print(f"为照片 {photo_path} 添加GPS信息失败: {e}")

    def process_photos(self, max_time_diff_minutes=5, interpolation_range_minutes=30):
        """
        处理文件夹中的所有照片

        Args:
            max_time_diff_minutes: 直接匹配的最大时间差（分钟）
            interpolation_range_minutes: 插值搜索范围（分钟）
        """
        if not self.load_gps_data():
            return

        supported_formats = (".jpg", ".jpeg", ".tiff", ".tif")
        processed_count = 0
        matched_count = 0

        for filename in os.listdir(self.photo_folder):
            if filename.lower().endswith(supported_formats):
                photo_path = os.path.join(self.photo_folder, filename)
                processed_count += 1

                print(f"\n处理照片: {filename}")

                # 获取照片时间
                photo_datetime = self.get_photo_datetime(photo_path)
                if photo_datetime is None:
                    print("无法获取照片时间信息")
                    continue

                print(f"照片拍摄时间（UTC）: {photo_datetime}")

                # 查找最接近的GPS坐标（包含插值）
                gps_coords = self.find_closest_gps(
                    photo_datetime, max_time_diff_minutes, interpolation_range_minutes
                )
                if gps_coords is None:
                    print("未找到匹配的GPS坐标")
                    continue

                latitude, longitude = gps_coords
                print(f"最终GPS坐标: {latitude:.6f}, {longitude:.6f}")

                # 添加GPS信息到照片
                self.add_gps_to_photo(photo_path, latitude, longitude)
                matched_count += 1

        print(
            f"\n处理完成! 共处理 {processed_count} 张照片，成功匹配 {matched_count} 张照片"
        )


def main():
    parser = argparse.ArgumentParser(description="为照片添加GPS地理位置信息")
    parser.add_argument("--csv", required=True, help="GPS轨迹CSV文件路径")
    parser.add_argument("--photos", required=True, help="照片文件夹路径")
    parser.add_argument(
        "--max-time-diff",
        type=int,
        default=5,
        help="直接匹配的最大时间差（分钟），默认5分钟",
    )
    parser.add_argument(
        "--interpolation-range",
        type=int,
        default=30,
        help="插值搜索范围（分钟），默认30分钟",
    )

    args = parser.parse_args()

    # 检查文件和文件夹是否存在
    if not os.path.exists(args.csv):
        print(f"CSV文件不存在: {args.csv}")
        return

    if not os.path.exists(args.photos):
        print(f"照片文件夹不存在: {args.photos}")
        return

    # 处理照片
    geotagger = GeotagPhotos(args.csv, args.photos)
    geotagger.process_photos(args.max_time_diff, args.interpolation_range)


if __name__ == "__main__":
    main()
