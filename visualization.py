import os
import re
import matplotlib.pyplot as plt
def is_extreme(value, threshold=80):
    return abs(value) > threshold

def calculate_derivative(values, timestamps):
    derivatives = []
    for i in range(len(values) - 1):
        dx = values[i + 1] - values[i]
        dt = timestamps[i + 1] - timestamps[i]
        if dt == 0:  # Avoid division by zero
            print(f"Timestamps at index {i} and {i+1} are equal! with value {timestamps[i]}")
            derivatives.append(derivatives[-1])  # Use the previous derivative value
            continue
        derivatives.append(dx / dt)
    derivatives.append(dx / dt)
    return derivatives

def read_pid_settings(file_path):
    pid_settings = {}
    with open(file_path, 'r') as file:
        file_content = file.read()
        pattern = r"Kp=([\d.]+), Ki=([\d.]+), Kd=([\d.]+)"
        matches = re.findall(pattern, file_content)
        pid_settings = [(float(kp), float(ki), float(kd)) for kp, ki, kd in matches]
    return pid_settings

def read_log(file_path):
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            segments = line.strip().split(";")
            parsed_values = [list(map(float, segment.strip("[]").split(","))) for segment in segments[2:]]
            
            time = float(segments[0].split(":")[1].strip())
            bat = int(segments[1].split(":")[1].strip())
            box = parsed_values[0]
            dif = parsed_values[1]
            ctrl = parsed_values[2]
            
            data.append([time, bat, box, dif, ctrl])

        return data
def sliding_window_average(values, window_size):
    extended_values = values + [0] * (window_size-1)
    return [sum(extended_values[i:i+window_size]) / window_size for i in range(len(values))]

if __name__ == "__main__":
    folder_dir = "20260824_143003"
    settings_file_path = os.path.join(folder_dir, 'settings.txt')
    log_file_path = os.path.join(folder_dir, 'log.txt')
    pid_values = read_pid_settings(settings_file_path)
    data = read_log(log_file_path)
    print(pid_values)
    print(data[0])

    time_values = []
    ud_dif_values = []
    fb_dif_values = []
    yv_dif_values = []
    rev_values = []
    fb_values = []
    ud_values = []
    yv_values = []
    bat_values = []
    overall_error_values = []
    
    fb_pos_values = []
    ud_pos_values = []
    yv_pos_values = []
    # loading & filtering data
    print(len(data))
    for i in range(len(data)):
        
        ud_dif = data[i][3][0] * 0.3
        fb_dif = data[i][3][1] * 3
        yv_dif = data[i][3][2] * 0.5

        rev_val = data[i][4][0]
        fb_val = -data[i][4][1]
        ud_val = -data[i][4][2]
        yv_val = data[i][4][3]
        
        
        if not is_extreme(ud_dif) and not is_extreme(fb_dif) and not is_extreme(yv_dif) and not is_extreme(ud_val) and not is_extreme(fb_val) and not is_extreme(yv_val):
            time_values.append(data[i][0] - data[0][0])
            bat_values.append(data[i][1])

            ud_dif_values.append(ud_dif)
            fb_dif_values.append(fb_dif)
            yv_dif_values.append(yv_dif)
            overall_error_values.append(ud_dif + fb_dif + yv_dif)
            
            rev_values.append(rev_val)
            fb_values.append(fb_val)
            ud_values.append(ud_val)
            yv_values.append(yv_val)

            if(len(fb_pos_values)>0):
                fb_pos_values.append(fb_pos_values[len(fb_pos_values)-1]+fb_val)
                ud_pos_values.append(ud_pos_values[len(ud_pos_values)-1]+ud_val)
                yv_pos_values.append(yv_pos_values[len(yv_pos_values)-1]+yv_val)
            else:
                fb_pos_values.append(0)
                ud_pos_values.append(0)
                yv_pos_values.append(0)
   
    window_size = 20
   
    # average data values
    # time_values = sliding_window_average(time_values, 1)
    ud_dif_values = sliding_window_average(ud_dif_values, window_size)
    fb_dif_values = sliding_window_average(fb_dif_values, window_size)
    yv_dif_values = sliding_window_average(yv_dif_values, window_size)
    fb_values = sliding_window_average(fb_values, window_size)
    ud_values = sliding_window_average(ud_values, window_size)
    yv_values = sliding_window_average(yv_values, window_size)
    overall_error_values = sliding_window_average(overall_error_values, window_size)

    # derivatives of the error values
    ud_dif_derivatives = calculate_derivative(ud_dif_values, time_values)
    fb_dif_derivatives = calculate_derivative(fb_dif_values, time_values)
    yv_dif_derivatives = calculate_derivative(yv_dif_values, time_values)
    overall_error_derivative = calculate_derivative(overall_error_values, time_values)

    ud_dif_derivatives = sliding_window_average(ud_dif_derivatives, window_size)
    fb_dif_derivatives = sliding_window_average(fb_dif_derivatives, window_size)
    yv_dif_derivatives = sliding_window_average(yv_dif_derivatives, window_size)
    overall_error_derivative = sliding_window_average(overall_error_derivative, window_size)
    
    # multiply k to derivatives
    ud_dif_derivatives = [x * 0.5 for x in ud_dif_derivatives]
    fb_dif_derivatives = [x * 0.5 for x in fb_dif_derivatives]
    yv_dif_derivatives = [x * 0.5 for x in yv_dif_derivatives]
    

    fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
         
    axs[0].plot(time_values, ud_dif_values, label=f"UD error", color='gray')
    axs[0].plot(time_values, fb_dif_derivatives, label="UD error Derivative")
    axs[0].plot(time_values, ud_values, label=f"UD speed control")
    axs[0].set_title(label=f'UD PID: Kp={pid_values[0][0]}, Ki={pid_values[0][1]}, Kd={pid_values[0][2]}')
    
    axs[1].plot(time_values, fb_dif_values, label=f"FB error", color='gray')
    axs[1].plot(time_values, ud_dif_derivatives, label="FB error Derivative")
    axs[1].plot(time_values, fb_values, label=f"FB speed control")
    axs[1].set_title(label=f'FB PID: Kp={pid_values[1][0]}, Ki={pid_values[1][1]}, Kd={pid_values[1][2]}')

    axs[2].plot(time_values, yv_dif_values, label=f"YV error", color='gray')
    axs[2].plot(time_values, yv_dif_derivatives, label="YV error Derivative")
    axs[2].plot(time_values, yv_values, label=f"YV speed control")
    axs[2].set_title(label=f'YV PID: Kp={pid_values[2][0]}, Ki={pid_values[2][1]}, Kd={pid_values[2][2]}')

    axs[3].plot(time_values, rev_values, label="REV speed")
    axs[3].plot(time_values, bat_values, label="battery")
    axs[3].set_xlabel('Time (s)')

    
    for ax in axs:
        # ax.set_xlim([0, 160])
        ax.legend()
        ax.grid(True)
        ax.tick_params(axis='x', which='both', labelsize=12)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    fig.suptitle('data visualization')
    plt.show()
    