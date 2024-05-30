import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def process_csv(file_path):
    # 读取CSV文件
    data = pd.read_csv(file_path)
    
    # 去除重复行
    data = data.drop_duplicates()
    
    # 将日期转换为Pandas的DateTime对象
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m')
    
    # 设置日期为索引
    data.set_index('Date', inplace=True)
    
    # 以每六个月为一个时间窗口计算EM的平均值
    six_month_avg = data['EM'].resample('6M').mean()
    
    # 将时间索引调整为每个时间段的最后一个月
    six_month_avg.index = six_month_avg.index - pd.offsets.MonthBegin(1)
    
    return six_month_avg

# 文件路径
monthly_average_em_path = 'monthly_average_em.csv'
lora_path = 'OCKL_result/lora.csv'
mixreview_path = 'OCKL_result/mixreview.csv'
vanilla_path = 'OCKL_result/vanilla.csv'

# 处理文件
monthly_average_em = process_csv(monthly_average_em_path)
lora = process_csv(lora_path)
mixreview = process_csv(mixreview_path)
vanilla = process_csv(vanilla_path)

# 绘制图形
plt.figure(figsize=(12, 8))
plt.plot(monthly_average_em.index.to_pydatetime(), monthly_average_em.values*100, marker='o', linewidth=4, label='Static RAG')
plt.plot(lora.index.to_pydatetime(), lora.values, marker='^', linewidth=4, label='Lora')
plt.plot(mixreview.index.to_pydatetime(), mixreview.values, marker='s', linewidth=4, label='Mix-Review')
plt.plot(vanilla.index.to_pydatetime(), vanilla.values, marker='x', linewidth=4, label='Vanilla CL')

plt.title('Six-Month Average EM Score Comparison', fontsize=22)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Average EM Score', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# 设置横坐标开始于2019年6月15日之前留有一些空余
plt.gca().set_xlim(left=pd.Timestamp('2019-06-15'))

# 设置纵坐标范围为0到30
plt.ylim(0, 30)

# 设置时间格式
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 取消网格
plt.grid(False)

# 添加图例
plt.legend(fontsize=14)

# 保存图形为PDF文件
plt.savefig('combined_six_month_average_em_score.pdf')

# 显示图形
plt.show()




# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates

# # 设置CSV文件的路径
# file_path = 'monthly_average_em.csv'

# # 读取CSV文件
# data = pd.read_csv(file_path)

# # 将日期转换为Pandas的DateTime对象
# data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m')

# # 设置日期为索引
# data.set_index('Date', inplace=True)

# # 以每六个月为一个时间窗口计算平均值
# six_month_avg = data.resample('6M').mean()

# # 将时间索引调整为每个时间段的最后一个月
# six_month_avg.index = six_month_avg.index - pd.offsets.MonthBegin(1)

# # 绘制图形，确保x轴和y轴的数据没有多维索引问题
# plt.figure(figsize=(10, 6))
# plt.plot(six_month_avg.index.to_pydatetime(), six_month_avg['EM'].values, marker='o', linewidth=2)  # 加粗线条

# # 设置标题和标签的字体大小
# plt.title('Six-Month Average EM Score at End of Period', fontsize=18)
# plt.xlabel('Date', fontsize=16)
# plt.ylabel('Average EM Score', fontsize=16)

# # 设置坐标轴刻度的字体大小
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

# # 设置横坐标开始于2019年7月之前留有一些空余
# plt.gca().set_xlim(left=pd.Timestamp('2019-06-15'))  # 从2019年6月15日开始，为了留出空余

# # 设置纵坐标范围为0到30
# plt.ylim(0, 0.30)

# # 设置时间格式
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# # 取消网格
# plt.grid(False)

# # 保存图形为PDF文件
# plt.savefig('six_month_average_em_score_adjusted.pdf')

# # 显示图形
# plt.show()
