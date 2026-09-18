import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    v_file = "data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset/V-M.csv"
    s_file = "data/raw/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-M.csv"

    v_df = pd.read_csv(v_file, encoding='latin1', on_bad_lines='skip')
    s_df = pd.read_csv(s_file, encoding='latin1', on_bad_lines='skip')

    v_time = v_df.iloc[:, 1]
    v_speed = v_df.iloc[:, 4] / 3.6 # km/h to m/s
    
    s_time = s_df.iloc[:, 7] / 1000.0 # ms to s
    s_accel_x = s_df.iloc[:, 9]
    s_accel_y = s_df.iloc[:, 10]
    s_accel_z = s_df.iloc[:, 11]
    s_accel_mag = np.sqrt(s_accel_x**2 + s_accel_y**2 + s_accel_z**2) - 9.8

    # A. Stationary Period (V_time around 50s-100s)
    plt.figure(figsize=(10,4))
    plt.plot(v_time, v_speed, label="GPS Speed (m/s)")
    plt.plot(s_time, s_accel_mag, label="IMU Accel Mag (m/s^2) - 9.8", alpha=0.5)
    plt.xlim(v_time.min(), v_time.min() + 100)
    plt.ylim(-5, 10)
    plt.legend()
    plt.title("A. Stationary Period")
    plt.savefig("reports/plot_A_stationary.png")
    
    # D. Low-Speed Region (0-5 m/s)
    plt.figure(figsize=(10,4))
    plt.plot(v_time, v_speed, label="GPS Speed (m/s)")
    plt.plot(s_time, s_accel_mag, label="IMU Accel Mag (m/s^2) - 9.8", alpha=0.5)
    plt.xlim(v_time.min() + 200, v_time.min() + 300)
    plt.ylim(-5, 15)
    plt.legend()
    plt.title("D. Low-Speed Region (0-5 m/s)")
    plt.savefig("reports/plot_D_low_speed.png")

    print("Generated plots in reports/")

if __name__ == "__main__":
    generate_plots()
