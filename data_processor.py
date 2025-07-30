# -*- coding: utf-8 -*-
"""
数据处理脚本
功能：
1. 遍历指定路径下的所有Excel文件（支持多层文件夹）
2. 将第一列时间拆分成：年、月、日、星期四列
3. 删除涨幅、振幅列中的百分号
4. 保持行顺序不变
5. 在项目中创建以今天日期命名的文件夹，保存处理后的文件
"""

import os
import pandas as pd
from datetime import datetime
import re
import shutil

# ==================== 配置区域 ====================
# 请在这里配置存放数据的绝对路径
DATA_ROOT_PATH = r"D:\code\Quantification"

# ==================== 配置区域结束 ====================

def get_weekday_number(weekday_chinese):
    """
    将中文星期转换为数字（1-7）
    """
    weekday_map = {
        '一': 1, '二': 2, '三': 3, '四': 4, 
        '五': 5, '六': 6, '日': 7, '天': 7
    }
    return weekday_map.get(weekday_chinese, 0)

def parse_time_column(time_str):
    """
    解析时间列，提取年、月、日、星期
    输入格式：2016-06-15,三
    输出：年、月、日、星期数字
    """
    try:
        # 分割日期和星期
        parts = time_str.split(',')
        if len(parts) != 2:
            return None, None, None, None
        
        date_part = parts[0].strip()
        weekday_part = parts[1].strip()
        
        # 解析日期
        date_obj = datetime.strptime(date_part, '%Y-%m-%d')
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        
        # 解析星期
        weekday_num = get_weekday_number(weekday_part)
        
        return year, month, day, weekday_num
    except Exception as e:
        print(f"解析时间失败: {time_str}, 错误: {e}")
        return None, None, None, None

def remove_percentage(value):
    """
    删除百分号并转换为数值
    """
    if pd.isna(value):
        return value

    if isinstance(value, str):
        # 移除百分号和加号
        cleaned = value.replace('%', '').replace('+', '')
        try:
            return float(cleaned)
        except ValueError:
            return value
    return value

def convert_to_number(value):
    """
    将包含逗号的文本数字转换为数值格式
    """
    if pd.isna(value):
        return value

    if isinstance(value, str):
        # 移除逗号分隔符
        cleaned = value.replace(',', '')
        try:
            # 尝试转换为整数
            if '.' not in cleaned:
                return int(cleaned)
            else:
                return float(cleaned)
        except ValueError:
            return value
    return value

def process_excel_file(file_path, output_path):
    """
    处理单个Excel文件
    """
    try:
        print(f"正在处理文件: {file_path}")

        # 读取Excel文件，尝试不同的引擎
        df = None
        engines = ['openpyxl', 'xlrd']

        for engine in engines:
            try:
                df = pd.read_excel(file_path, engine=engine)
                break
            except:
                continue

        # 如果所有引擎都失败，尝试作为CSV读取
        if df is None:
            try:
                df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            except:
                try:
                    df = pd.read_csv(file_path, sep='\t', encoding='gbk')
                except Exception as e:
                    print(f"文件读取失败: {e}")
                    return False
        
        # 检查是否有时间列（假设第一列是时间列）
        if df.empty or len(df.columns) == 0:
            print(f"文件为空或无列: {file_path}")
            return False

        # 先检查第一条数据的时间，决定过滤策略
        first_row_time = df.iloc[0, 0] if len(df) > 0 else None
        if first_row_time and not pd.isna(first_row_time):
            try:
                # 解析第一条数据的时间
                time_parts = str(first_row_time).split(',')
                if len(time_parts) >= 1:
                    date_part = time_parts[0].strip()
                    first_date = pd.to_datetime(date_part)
                    cutoff_date = pd.to_datetime('2015-11-01')

                    if first_date < cutoff_date:
                        use_date_filter = True
                        skip_rows = False
                    else:
                        use_date_filter = False
                        skip_rows = True
                else:
                    use_date_filter = False
                    skip_rows = True
            except Exception as e:
                use_date_filter = False
                skip_rows = True
        else:
            use_date_filter = False
            skip_rows = True

        # 根据策略处理数据
        if skip_rows:
            if len(df) > 50:
                df = df.iloc[50:].reset_index(drop=True)

        # 获取第一列的列名
        time_column = df.columns[0]
        
        # 解析时间列，创建新的列
        years = []
        months = []
        days = []
        weekdays = []
        
        for idx, time_value in df[time_column].items():
            if pd.isna(time_value):
                years.append(None)
                months.append(None)
                days.append(None)
                weekdays.append(None)
            else:
                year, month, day, weekday = parse_time_column(str(time_value))
                years.append(year)
                months.append(month)
                days.append(day)
                weekdays.append(weekday)
        
        # 创建新的DataFrame，保持原有顺序
        new_df = df.copy()

        # 删除第一列时间（因为已经拆分了）
        new_df = new_df.drop(columns=[time_column])

        # 在开头插入新的时间列
        new_df.insert(0, '年', years)
        new_df.insert(1, '月', months)
        new_df.insert(2, '日', days)
        new_df.insert(3, '星期', weekdays)

        # 删除所有Unnamed列（通常是空列导致的）
        unnamed_cols = [col for col in new_df.columns if 'Unnamed' in str(col)]
        if unnamed_cols:
            new_df = new_df.drop(columns=unnamed_cols)
        
        # 处理涨幅和振幅列，删除百分号
        for col in new_df.columns:
            if '涨幅' in col or '振幅' in col:
                new_df[col] = new_df[col].apply(remove_percentage)

        # 处理总手和金额列，转换为数字格式
        for col in new_df.columns:
            if '总手' in col or '金额' in col:
                new_df[col] = new_df[col].apply(convert_to_number)

        # 根据策略决定是否进行日期过滤
        if use_date_filter:
            # 过滤数据：只保留2016年2月之后的数据
            date_filter = (
                (new_df['年'] > 2016) |  # 2016年之后的所有年份
                ((new_df['年'] == 2016) & (new_df['月'] >= 2))  # 2016年2月及之后
            )

            new_df = new_df[date_filter].reset_index(drop=True)

            if len(new_df) == 0:
                print("警告：过滤后没有数据")
                return False
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 根据原文件扩展名决定保存格式，但统一保存为.xlsx格式
        if output_path.lower().endswith('.xls'):
            output_path = output_path[:-4] + '.xlsx'
        elif not output_path.lower().endswith('.xlsx'):
            output_path = output_path + '.xlsx'

        # 保存处理后的文件
        try:
            new_df.to_excel(output_path, index=False, engine='openpyxl')
            return True
        except Exception as e:
            print(f"保存Excel文件失败: {e}")
            # 尝试保存为CSV文件作为备选
            csv_path = output_path.replace('.xlsx', '.csv')
            try:
                new_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                return True
            except Exception as csv_e:
                print(f"保存文件失败: {csv_e}")
                return False
        
    except Exception as e:
        print(f"处理文件失败: {file_path}, 错误: {e}")
        return False

def find_excel_files(root_path):
    """
    递归查找所有Excel文件，但排除processed_data开头的文件夹
    """
    excel_files = []

    for root, dirs, files in os.walk(root_path):
        # 排除processed_data开头的文件夹
        dirs[:] = [d for d in dirs if not d.startswith('processed_data')]

        for file in files:
            if file.lower().endswith(('.xlsx', '.xls')):
                file_path = os.path.join(root, file)
                excel_files.append(file_path)

    return excel_files

def create_output_structure(input_path, root_input_path, root_output_path):
    """
    根据输入路径创建对应的输出路径结构
    """
    # 获取相对路径
    rel_path = os.path.relpath(input_path, root_input_path)
    
    # 创建输出路径
    output_path = os.path.join(root_output_path, rel_path)
    
    return output_path

def main():
    """
    主函数
    """
    print("=" * 50)
    print("数据处理脚本启动")
    print("=" * 50)
    
    # 检查数据根路径是否存在
    if not os.path.exists(DATA_ROOT_PATH):
        print(f"错误：数据根路径不存在: {DATA_ROOT_PATH}")
        return
    
    # 创建输出文件夹（以今天日期命名）
    today = datetime.now().strftime("%Y-%m-%d")
    output_root = os.path.join(os.getcwd(), f"processed_data_{today}")
    
    print(f"数据根路径: {DATA_ROOT_PATH}")
    print(f"输出根路径: {output_root}")
    
    # 查找所有Excel文件
    print("\n正在搜索Excel文件...")
    excel_files = find_excel_files(DATA_ROOT_PATH)
    
    if not excel_files:
        print("未找到任何Excel文件")
        return
    
    print(f"找到 {len(excel_files)} 个Excel文件")
    
    # 处理每个文件
    success_count = 0
    fail_count = 0
    
    for file_path in excel_files:
        # 创建对应的输出路径
        output_path = create_output_structure(file_path, DATA_ROOT_PATH, output_root)
        
        # 处理文件
        if process_excel_file(file_path, output_path):
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "=" * 50)
    print("处理完成")
    print(f"成功处理: {success_count} 个文件")
    print(f"处理失败: {fail_count} 个文件")
    print(f"输出目录: {output_root}")
    print("=" * 50)

if __name__ == "__main__":
    main()
