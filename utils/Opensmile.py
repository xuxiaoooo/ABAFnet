# import os
# import subprocess
# from pydub import AudioSegment
# from glob import glob

# def extract_emolarge_features(input_file, output_file, opensmile_path, config_path):
#     command = f"{opensmile_path} -C {config_path} -I {input_file} -O {output_file}"
#     subprocess.call(command, shell=True)

# def main():
#     input_folder = '/home/user/xuxiao/DeepL/zxxaudio'
#     output_folder = '/home/user/xuxiao/DeepL/zxx-opensmilehandle'
#     opensmile_path = '/home/user/xuxiao/DeepL/opensmile-3.0.1-linux-x64/bin/SMILExtract'
#     config_path = '/home/user/xuxiao/DeepL/opensmile-3.0.1-linux-x64/config/misc/emo_large.conf'

#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     audio_files = glob(os.path.join(input_folder, '*.wav'))

#     for audio_file in audio_files:
#         file_name = os.path.splitext(os.path.basename(audio_file))[0]
#         wav_file = os.path.join(input_folder, f"{file_name}.wav")

#         output_file = os.path.join(output_folder, f"{file_name}.csv")
#         extract_emolarge_features(wav_file, output_file, opensmile_path, config_path)

# if __name__ == "__main__":
#     main()


import os
import csv
from glob import glob

def process_csv_file(file_path, file_name):
    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        data = [row for row in csv_reader]

    header = [col[0].split(" ")[1] for col in data[2:6556] if len(col) > 0 and " " in col[0]]
    header.insert(0, "name")
    values = data[6559][:-1]  # 去掉最后一列class
    values.insert(0, file_name)

    return header, values

def main():
    input_folder = '/home/user/xuxiao/DeepL/zxx-opensmilehandle'
    output_file = os.path.join(input_folder, "zxx_emolarge_features.csv")

    csv_files = glob(os.path.join(input_folder, '*.csv'))

    all_data = []
    for i, csv_file in enumerate(csv_files):
        file_name = os.path.splitext(os.path.basename(csv_file))[0]

        header, values = process_csv_file(csv_file, file_name)
        all_data.append(values)

        # 只获取一次列名
        if i == 0:
            final_header = header

    # 写入汇总CSV文件
    with open(output_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(final_header)
        csv_writer.writerows(all_data)

if __name__ == "__main__":
    main()