import numpy as np
import matplotlib.pyplot as plt
import librosa

# 读取音频文件
y, sr = librosa.load('/home/user/xuxiao/DeepL/zxxaudio/6960.wav')
start_time = 5
end_time = 15
start_frame = start_time * sr
end_frame = end_time * sr
y = y[start_frame:end_frame]
# 获取时间信息
time = np.arange(0, len(y)) / sr

# 获取振幅信息
amplitude = y

# 将音频分成7份
num_sections = 7
section_len = len(y) // num_sections
sections = [y[i*section_len:(i+1)*section_len] for i in range(num_sections)]

color1 = (98 / 255, 46 / 255, 83 / 255)
color2 = (128 / 255, 130 / 255, 170 / 255)

# 为奇数和偶数部分设置不同的颜色
colors = [color1 if i % 2 == 0 else color2 for i in range(num_sections)]

# 创建图形和轴
fig, ax = plt.subplots()

# 绘制波形图
for i, section in enumerate(sections):
    section_time = time[i*section_len:(i+1)*section_len]
    section_color = colors[i]
    for j in range(1, len(section)):
        ax.plot(section_time[j-1:j+1], section[j-1:j+1], color=section_color)

# 设置横纵坐标标签
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')

# 显示图形
plt.savefig('a.png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
