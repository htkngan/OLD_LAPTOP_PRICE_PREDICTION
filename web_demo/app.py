from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('./save_model/gradient_boosting_model.joblib')

graphic_card_type_map = {"Intel Graphics": 1, "NVIDIA Graphics": 3, "AMD Graphics": 0, "Microsoft SQ1 Adreno": 2}
cpu_type_map = {'Intel Core i5':4, 'Intel Core i3':3, 'Intel Core i7':5, 'AMD Ryzen 9':2, 'AMD Ryzen 7':1, 'AMD Ryzen 5':0, 'Intel Pentium':6}
laptop_state_map = {'Cũ đẹp': 0, 'Trầy xước':1, 'Đã kích hoạt':3, 'Xước cấn':2}
series_map = {'Office':2, '2 in 1':0, 'Ultrabook':3, 'Gaming':1}
screen_tech_map = {'LED BACKLIT':5, 'PIXELSENSE':8, 'IPS':3, 'DOLBY VISION':1, 'NTSC':6, 'OLED':7, 'HDR':2, 'TN':9, 'LED':4, 'COMFYVIEW':0}
cell_num_map = {'2 cell':0, '3 cell':1, '4 cell':2, '6 cell':3}
ram_mem_map = {"8GB": 5, "16GB": 1, "4GB": 4, "12GB": 0, "32GB": 3, "24GB": 2}
ram_type_map = {'LPDDR4X':4, 'DDR4':1, 'LPDDR5':5, 'DDR5':2, 'LPDDR4':3, 'DDR IV':0}
hard_disk_map = {'SSD 512GB':3, 'SSD 128GB':0, 'SSD 256GB':2, 'SSD 1TB':1}
intel_tech_map = {'Không':2, 'Intel Evo':0, 'Intel Gaming':1}
screen_resolution_map = {'(1920, 1080)':3, '(2736, 1824)':10, '(1920, 1200)':4, '(2880, 1920)':12,
       '(2240, 1400)':6, '(1536, 1024)':2, '(2880, 1800)':11, '(2560, 1600)':9,
       '(2560, 1440)':8, '(2160, 1440)':5, '(1280, 720)':0, '(1366, 768)':1,
       '(3840, 2160)':14, '(2256, 1504)':7, '(3000, 2000)':13}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        battery_capacity = float(request.form['battery_capacity'])
        graphic_card_type = float(graphic_card_type_map[request.form['graphic_card_type']])
        ram_mem = float(ram_mem_map[request.form['ram_mem']])
        laptop_state = float(laptop_state_map[request.form['laptop_state']])
        cpu_type = float(cpu_type_map[request.form['cpu_type']])
        series = float(series_map[request.form['series']])
        screen_tech = float(screen_tech_map[request.form['screen_tech']])
        intel_tech = float(intel_tech_map[request.form['intel_tech']])
        cell_num = float(cell_num_map[request.form['cell_num']])
        ram_type = float(ram_type_map[request.form['ram_type']])
        hard_disk = float(hard_disk_map[request.form['hard_disk']])
        screen_resolution = float(screen_resolution_map[request.form['screen_resolution']])

        input_data = np.array([[battery_capacity, graphic_card_type, cpu_type, laptop_state, series, screen_tech, cell_num, ram_mem, ram_type, hard_disk, intel_tech, screen_resolution]])
        prediction = model.predict(input_data)
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)