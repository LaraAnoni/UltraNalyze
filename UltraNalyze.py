import customtkinter as ctk
from tkinter import ttk, Tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks 
from PIL import Image, ImageTk 
import pandas as pd
import os
import sys
from PIL import Image, ImageTk
from tkinter import ttk, PhotoImage
import csv
from datetime import datetime
from scipy.signal import savgol_filter
import pywt
from PyEMD import EMD
from scipy.signal import hilbert
from scipy.integrate import trapz

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Variáveis globais
signal = None
acquisition_frequency = None
calibration_offset = None
found_harmonics = None
elements_to_remove = []


# Função para carregar o arquivo
def load_file():
    global signal, acquisition_frequency, calibration_offset, normalize_var
    try:
        calibration_offset = float(calibration_offset_entry.get())
        acquisition_frequency = float(acquisition_frequency_entry.get())
        start_hanning = int(start_hanning_entry.get())
        end_hanning = int(end_hanning_entry.get())
        normalize = normalize_var.get()

        filepath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if filepath:
            print("Arquivo carregado:", filepath)
            process_signal(filepath, calibration_offset, acquisition_frequency, normalize)
    except ValueError:
        messagebox.showerror("Erro", "Por favor, insira valores válidos para Offset de Calibração e Frequência de Aquisição.")

# Função para processar o sinal
def process_signal(filepath, calibration_offset, acquisition_frequency, normalize):
    global signal
    signal = np.loadtxt(filepath)

    if normalize:
        signal = signal / np.max(np.abs(signal))  # Normalização simples [-1, 1]

    time = (np.arange(len(signal)) / (acquisition_frequency * 1e3))

    # Aplicar o Offset de Calibração deslocando o tempo
    time += calibration_offset*1e-6

    N = len(signal)
    T = (1.0 / (acquisition_frequency * 1e3))*1e-3
    if apply_hanning_var.get():
        start_hanning = int(start_hanning_entry.get())
        end_hanning = int(end_hanning_entry.get())
        # Verificações básicas
        if start_hanning < 0 or end_hanning > N or start_hanning >= end_hanning:
            messagebox.showerror("Erro", "Intervalo da janela de Hanning inválido.")
            return
        # Definir o intervalo Hanning Window
        window_length = end_hanning - start_hanning
        window = np.hanning(window_length)
        # Criar um sinal ajustado onde o intervalo fora da janela é zerado
        windowed_signal = np.zeros_like(signal)  # Sinal zerado
        windowed_signal[start_hanning:end_hanning] = signal[start_hanning:end_hanning] * window  # Aplicar a janela entre os pontos 150 e 750
    
        # Calcular a FFT do sinal com janela parcial
        yf = fft(windowed_signal)
        xf = fftfreq(N, T)[:N//2]

    else:
        # Calcular a FFT do sinal com janela parcial
        yf = fft(signal)
        xf = fftfreq(N, T)[:N//2]
    
    update_plots(time, signal, xf, yf)

# Função para tratar sinais que vêm direto do ensaio
def process_RAWsignals(threshold=2):
    input_excels = filedialog.askopenfilenames(
        title="Select Data File",
        filetypes=[("Data files", "*.xlsx *.xls *.csv")]
    )

    if not input_excels:
        print("No files selected.")
        return
    
    for input_excel in input_excels:

        if input_excel.endswith('.csv'):
            df_raw = pd.read_csv(
                input_excel,
                header=None,
                skiprows=17,
                sep=None,          # mantenha como estava funcionando no Excel
                encoding='utf-16',
                engine='python'
            )

            # Remove colunas A–W
            df_raw = df_raw.iloc[:, 23:]

            # Converte tudo para número
            df_raw = df_raw.apply(pd.to_numeric, errors='coerce')


            # Salva XLS temporário
            temp_xls = os.path.splitext(input_excel)[0] + "_cleaned.xlsx"
            df_raw.to_excel(temp_xls, header=False, index=False)

            input_excel = temp_xls  # 🔑 redireciona o fluxo

        
        if not (input_excel.endswith('.xls') or input_excel.endswith('.xlsx')):
            print("Invalid file format. Please select a .xls or .xlsx file.")
            return
        
        current_directory = os.path.dirname(os.path.abspath(__file__))
        input_filename = os.path.splitext(os.path.basename(input_excel))[0]
        output_folder = os.path.join(current_directory, f"{input_filename}_time-domain")
        FFToutput_folder = os.path.join(current_directory, f"{input_filename}_f-domain")

        nleituras = simpledialog.askinteger("Number of readings", "Insert number of readings in each measurement:")
        # nleituras = 10

        if not nleituras or nleituras <= 0:
            messagebox.showerror("Error", "Invalid number of readings. Try again.")
            return

        if input_excel.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(input_excel, header=None)
        else:
            messagebox.showerror("Error", "Unsupported file format.")
            return
        
        print("Número de linhas após limpeza:", df.shape[0])
        print("Número de colunas após limpeza:", df.shape[1])

        num_measurements = df.shape[0] // nleituras
        print("Número de medições:", num_measurements)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if not os.path.exists(FFToutput_folder):
            os.makedirs(FFToutput_folder)

        # Pergunta sobre a Hanning window antes do loop
        apply_hanning = messagebox.askyesno("Apply Hanning Window", "Do you want to apply a Hanning window before FFT?")
        apply_hanning = False
        start, end = None, None

        if apply_hanning:
            root = Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            start = simpledialog.askinteger("Start", "Enter the start index:", parent=root)
            end = simpledialog.askinteger("End", "Enter the end index:", parent=root)
            root.destroy()

            # Valida os valores inseridos
            if start is None or end is None or start >= end or start < 0:
                messagebox.showerror("Error", "Invalid start or end values. FFT will be computed without Hanning window.")
                apply_hanning = False  # Desativa se for inválido

        for i in range(num_measurements):
            signals = df.iloc[i * nleituras:(i + 1) * nleituras].values
            initial_mean = np.mean(signals, axis=0)
            distances = np.linalg.norm(signals - initial_mean, axis=1)
            std_dev = np.std(distances)
            filtered_signals = signals[distances < threshold * std_dev]

            if filtered_signals.size == 0:
                final_mean_signal = initial_mean
                print(f"Aviso: Todos os sinais no grupo {i + 1} foram filtrados.")
            else:
                final_mean_signal = np.mean(filtered_signals, axis=0)

            def moving_average(signal, window_size):
                return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')

            smoothed_signal = moving_average(final_mean_signal, window_size=5)

            output_filename = os.path.join(output_folder, f"signal_{i + 1}.txt")
            np.savetxt(output_filename, smoothed_signal)
            print(f"Sinal {i + 1} processado e salvo em {output_filename}")

            acquisition_frequency = float(acquisition_frequency_entry.get())
            N = len(smoothed_signal)
            T = 1.0 / (acquisition_frequency * 1e6)

            if apply_hanning and end <= N:
                window_length = end - start
                window = np.hanning(window_length)
                windowed_signal = np.zeros_like(smoothed_signal)
                windowed_signal[start:end] = smoothed_signal[start:end] * window
            else:
                windowed_signal = smoothed_signal  # Usa o sinal original se a janela não for aplicada

            # yf = fft(windowed_signal)
            # xf = fftfreq(N, T)[:N // 2]
            # yf_abs = np.abs(yf[:N // 2])

            # FFToutput_filename = os.path.join(FFToutput_folder, f"FFT_{i + 1}.txt")
            # np.savetxt(FFToutput_filename, np.column_stack((xf, yf_abs)))
            # print(f"FFT {i + 1} processado e salvo em {FFToutput_filename}")

        # messagebox.showinfo("Success", "Processing completed!")
        print("Processing completed!")

# Função para atualizar os gráficos
def update_plots(time, signal, xf, yf):
    global ax1, ax2
    # Limpar gráficos anteriores
    ax1.clear()
    ax2.clear()

    # Plotar o sinal no domínio do tempo
    ax1.plot(time*1e6, signal)
    ax1.set_title("Sinal no Domínio do Tempo")
    ax1.set_xlabel("Tempo (μs)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)  # Adiciona a grade ao gráfico de tempo

    # Plotar a FFT do sinal
    ax2.plot(xf*1e-6, 2.0/len(signal) * np.abs(yf[:len(signal)//2]))
    ax2.set_title("FFT do Sinal")
    ax2.set_xlabel("Frequência (kHz)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True)  # Adiciona a grade ao gráfico de FFT

    # Atualizar a tela com os novos gráficos
    canvas.draw()

def find_harmonics():
    global found_harmonics
    if signal is None or acquisition_frequency is None:
        messagebox.showwarning("Aviso", "Carregue um sinal e defina a frequência de aquisição antes de encontrar harmônicos.")
        return
    
    try:
        central_freq_str = float(centralFreq.get())
        if not central_freq_str:
            messagebox.showwarning("Aviso", "Por favor, insira a frequência central.")
            return
        
        central_freq = float(central_freq_str) * 1e6 

        N = len(signal)
        T = 1.0 / (acquisition_frequency * 1e6)

        if apply_hanning_var.get():
            start_hanning = int(start_hanning_entry.get())
            end_hanning = int(end_hanning_entry.get())
            # Verificações básicas
            if start_hanning < 0 or end_hanning > N or start_hanning >= end_hanning:
                messagebox.showerror("Erro", "Intervalo da janela de Hanning inválido.")
                return
            # Definir o intervalo Hanning Window
            window_length = end_hanning - start_hanning
            window = np.hanning(window_length)
            # Criar um sinal ajustado onde o intervalo fora da janela é zerado
            windowed_signal = np.zeros_like(signal)  # Sinal zerado
            windowed_signal[start_hanning:end_hanning] = signal[start_hanning:end_hanning] * window  # Aplicar a janela entre os pontos 150 e 750
    
            # Calcular a FFT do sinal com janela parcial
            yf = fft(windowed_signal)
            xf = fftfreq(N, T)[:N//2]

        else:
            # Calcular a FFT do sinal com janela parcial
            yf = fft(signal)
            xf = fftfreq(N, T)[:N//2]
        

        harmonic_frequencies = []
        harmonic_amplitudes = []

        # Procurar picos ao redor da frequência central e seus harmônicos
        for i in range(1, 5):  # Consideramos os primeiros 4 harmônicos
            if i == 1:
                target_freq = i * central_freq
                tolerance = 0.2 * central_freq  # Tolerância de 20% ao redor da frequência
            else:
                target_freq = i * central_freq_real
                tolerance = 0.3 * central_freq_real

            # Encontrar os índices dentro da faixa desejada
            valid_indices = np.where((xf >= target_freq - tolerance) & (xf <= target_freq + tolerance))[0]

            if valid_indices.size > 0:  # Se existir pelo menos um valor dentro do intervalo
                best_index = max(valid_indices, key=lambda k: np.abs(yf[k]))  # Maior amplitude dentro da faixa

                harmonic_frequencies.append(xf[best_index])
                harmonic_amplitudes.append((2.0/N) * np.abs(yf[best_index]))

                if i == 1:  
                    central_freq_real = xf[best_index]  # Atualizar a frequência real do primeiro harmônico


        # Exibir os resultados no harmonic_frame
        result_textbox.delete(1.0, "end")  # Limpar resultados anteriores
        result_textbox.insert("end", "Frequência Fundamental e Harmônicos:\n")
        for i, freq in enumerate(harmonic_frequencies):
            # Calcular e escrever o valor de beta
            if len(harmonic_amplitudes) > 1:  # Verificar se há ao menos 2 harmônicos para calcular beta
                A1 = harmonic_amplitudes[0]  # Amplitude do primeiro harmônico (fundamental)
                A2 = harmonic_amplitudes[1]  # Amplitude do segundo harmônico (índice 2)
                A3 = harmonic_amplitudes[2]

                if A1 != 0:
                    beta = A2 / (A1 ** 2)
                    gama = A3 / (A1 ** 3)
                    result_textbox.insert("end", f"Beta: {beta:.3f}\n")
                    result_textbox.insert("end", f"Gama: {gama:.3f}\n")
                else:
                    result_textbox.insert("end", f"Beta e gama: {float('nan')}\n")
            else:
                result_textbox.insert(0)
            result_textbox.insert("end", f"H{i+1}: {freq/1e6:.2f} kHz, Amplitude: {harmonic_amplitudes[i]:.3f}\n")
        
        # Salvar as frequências e amplitudes para visualização
        found_harmonics = (harmonic_frequencies, harmonic_amplitudes)
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro na análise harmônica: {str(e)}")

def calculate_energy_distribution():
    global found_harmonics
    if signal is None or acquisition_frequency is None:
        messagebox.showwarning("Aviso", "Carregue um sinal e defina a frequência de aquisição antes de calcular a distribuição de energia.")
        return
    
    try:
        central_freq_str = centralFreq.get()
        if not central_freq_str:
            messagebox.showwarning("Aviso", "Por favor, insira a frequência central.")
            return
        
        central_freq = float(central_freq_str) * 1e6 

        N = len(signal)
        T = 1.0 / (acquisition_frequency * 1e6)

        if apply_hanning_var.get():
            start_hanning = int(start_hanning_entry.get())
            end_hanning = int(end_hanning_entry.get())
            # Verificações básicas
            if start_hanning < 0 or end_hanning > N or start_hanning >= end_hanning:
                messagebox.showerror("Erro", "Intervalo da janela de Hanning inválido.")
                return
            # Definir o intervalo Hanning Window
            window_length = end_hanning - start_hanning
            window = np.hanning(window_length)
            # Criar um sinal ajustado onde o intervalo fora da janela é zerado
            windowed_signal = np.zeros_like(signal)  # Sinal zerado
            windowed_signal[start_hanning:end_hanning] = signal[start_hanning:end_hanning] * window  # Aplicar a janela entre os pontos 150 e 750
    
            # Calcular a FFT do sinal com janela parcial
            yf = fft(windowed_signal)
            xf = fftfreq(N, T)[:N//2]
            
        else:
            # Calcular a FFT do sinal com janela parcial
            yf = fft(signal)
            xf = fftfreq(N, T)[:N//2]

        # Encontrar o pico máximo
        max_amp = np.max(np.abs((2.0/N) * np.abs(yf[:N//2])))
        # Encontrar picos na FFT
        peaks, _ = find_peaks((2.0/N) * np.abs(yf[:N//2]), height=0.05*max_amp)  # Ajustar o 'height' conforme necessário
        
        # Encontrar pico máximo ao redor da frequência central
        tolerance = 0.3 * central_freq  # Tolerância de 30% ao redor da frequência central
        max_amplitude = 0
        central_peak = None

        for peak in peaks:
            if abs(xf[peak] - central_freq) <= tolerance:
                amplitude = (2.0/N) * np.abs(yf[peak])
                if amplitude > max_amplitude:
                    max_amplitude = amplitude
                    central_peak = peak
        result_textbox.insert("end", f"Pico {central_peak:.2f} kHz")
        
        if central_peak is None:
            messagebox.showwarning("Aviso", "Não foi encontrado um pico significativo ao redor da frequência central.")
            return
        
        # Calcular a energia na banda central (central_freq ± 5 kHz)
        central_band = (xf >= (xf[central_peak] - 5* 1e6)) & (xf <= (xf[central_peak] + 5* 1e6))
        yf_central = yf[:N//2]
        central_energy = np.sum(((2.0/N)*np.abs(yf_central[central_band]))**2)
        
        # Calcular a energia nas bandas laterais (50 kHz de cada lado)
        left_band = (xf >= (xf[central_peak] - 55* 1e6)) & (xf < (xf[central_peak] - 5* 1e6))
        right_band = (xf > (xf[central_peak] + 5* 1e6)) & (xf <= (xf[central_peak] + 55* 1e6))
        #print(f"Tamanho de central_band: {((2.0/N)*np.abs(yf_central[left_band]))}")
        left_energy = np.sum(((2.0/N)*np.abs(yf_central[left_band]))**2)
        right_energy = np.sum(((2.0/N)*np.abs(yf_central[right_band]))**2)
        
        # Calcular a distribuição de energia
        if central_energy != 0:
                energy_distribution = (left_energy + right_energy)/central_energy
        else:
            energy_distribution=0
            messagebox.showwarning("Aviso", "Energia zero encontrada nas bandas central, não é possível calcular a distribuição de energia.")
            return
        
        # Plotar na FFT e marcar as regiões das 3 bandas
        band1=ax2.axvspan((xf[central_peak] - 5* 1e6)/1e6, (xf[central_peak] + 5* 1e6)/1e6, color='green', alpha=0.3, label='Banda Central ±5 kHz')
        band2=ax2.axvspan((xf[central_peak] - 55* 1e6)/1e6, (xf[central_peak] - 5* 1e6)/1e6, color='red', alpha=0.3, label='Banda Esquerda 20 kHz')
        band3=ax2.axvspan((xf[central_peak] + 5* 1e6)/1e6, (xf[central_peak] + 55* 1e6)/1e6, color='blue', alpha=0.3, label='Banda Direita 20 kHz')
        legband=ax2.legend()
        canvas.draw()
        elements_to_remove.append(band1)
        elements_to_remove.append(band2)
        elements_to_remove.append(band3)
        elements_to_remove.append(legband)

        result_textbox.delete(1.0, "end")  # Limpar resultados anteriores
        result_textbox.insert("end", f"Distribuição de Energia: {energy_distribution:.2f}\n")
        result_textbox.insert("end", f"Energia da Banda Central: {central_energy:.2e}\n")
        result_textbox.insert("end", f"Energia da Banda Esquerda: {left_energy:.2e}\n")
        result_textbox.insert("end", f"Energia da Banda Direita: {right_energy:.2e}\n")

    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro no cálculo da distribuição de energia: {str(e)}")

def plot_harmonics():
    if found_harmonics:
        harmonic_frequencies, _ = found_harmonics
        for freq in harmonic_frequencies:
            lines=ax2.axvline(x=freq/1e6, color='r', linestyle='--')
            elements_to_remove.append(lines)
        canvas.draw()
    else:
        messagebox.showwarning("Aviso", "Por favor, encontre os harmônicos antes de plotar.")

def SPC():
    global signal, acquisition_frequency
    if signal is None or acquisition_frequency is None:
        messagebox.showwarning("Aviso", "Carregue um sinal e defina a frequência de aquisição antes de realizar a análise SPC.")
        return

    try:
        N = len(signal)
        T = 1.0 / (acquisition_frequency * 1e6)

        if apply_hanning_var.get():
            start_hanning = int(start_hanning_entry.get())
            end_hanning = int(end_hanning_entry.get())
            # Verificações básicas
            if start_hanning < 0 or end_hanning > N or start_hanning >= end_hanning:
                messagebox.showerror("Erro", "Intervalo da janela de Hanning inválido.")
                return
            # Definir o intervalo Hanning Window
            window_length = end_hanning - start_hanning
            window = np.hanning(window_length)
            # Criar um sinal ajustado onde o intervalo fora da janela é zerado
            windowed_signal = np.zeros_like(signal)  # Sinal zerado
            windowed_signal[start_hanning:end_hanning] = signal[start_hanning:end_hanning] * window  # Aplicar a janela entre os pontos 150 e 750
    
            # Calcular a FFT do sinal com janela parcial
            yf = fft(windowed_signal)
            xf = fftfreq(N, T)[:N//2]
            
        else:
            # Calcular a FFT do sinal com janela parcial
            yf = fft(signal)
            xf = fftfreq(N, T)[:N//2]

        # Encontrar o pico máximo
        max_amplitude = np.max(np.abs((2.0/N) * np.abs(yf[:N//2])))

        # Encontrar picos na FFT acima do threshold
        peaks, _ = find_peaks((2.0/N) * np.abs(yf[:N//2]), height=0.05*max_amplitude)

        # Somar amplitudes dos picos
        peak_amplitudes = (2.0/N) * np.abs(yf[peaks])
        total_amplitude = np.sum(peak_amplitudes)

        # Calcular média das amplitudes
        spi = total_amplitude / max_amplitude
        
        # Exibir resultados
        resultSPC_textbox.delete(1.0, "end")
        resultSPC_textbox.insert("end", f"SPI: {spi}\n")

    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro na análise SPC: {str(e)}")

def HHT():
    global signal, acquisition_frequency

    if signal is None or acquisition_frequency is None:
        messagebox.showwarning("Aviso", "Carregue um sinal e defina a frequência de aquisição antes de aplicar a HHT.")
        return

    try:
        N = len(signal)
        T = 1.0 / (acquisition_frequency * 1e6)
        t = np.linspace(0, N*T, N)

        # Aplicar Hanning, se for o caso
        if apply_hanning_var.get():
            start_hanning = int(start_hanning_entry.get())
            end_hanning = int(end_hanning_entry.get())
            if start_hanning < 0 or end_hanning > N or start_hanning >= end_hanning:
                messagebox.showerror("Erro", "Intervalo da janela de Hanning inválido.")
                return

            window_length = end_hanning - start_hanning
            window = np.hanning(window_length)
            windowed_signal = np.zeros_like(signal)
            windowed_signal[start_hanning:end_hanning] = signal[start_hanning:end_hanning] * window
            signal_to_analyze = windowed_signal
        else:
            signal_to_analyze = signal

       # Aplicar Hilbert no sinal diretamente
        analytic_signal = hilbert(signal_to_analyze)
        amplitude_envelope = np.abs(analytic_signal)

        # Cálculo da área sob o envelope
        area_envelope = trapz(amplitude_envelope, t)

        #Para a frequnecia
        yf = fft(signal_to_analyze)
        xf = fftfreq(N, T)[:N // 2] 
        magnitude_spectrum = 2.0 / N * np.abs(yf[:N // 2])  # módulo da FFT = envelope espectral
        area_spectral_envelope = trapz(magnitude_spectrum, xf)

        # Exibir resultados
        resultSPC_textbox.delete(1.0, "end")
        resultSPC_textbox.insert("end", f"Envelope area: {area_envelope*1e6}\n")
        resultSPC_textbox.insert("end", f"Envelope FFT area: {area_spectral_envelope*1e-3}\n")

        plt.close('all')

        # Plotar sinal original + envelope
        '''plt.figure(figsize=(12, 5))
        plt.plot(t*1e6, signal_to_analyze, label='Sinal Original', color='black')
        plt.plot(t*1e6, amplitude_envelope, label='Envelope Superior', color='red')
        plt.plot(t*1e6, -amplitude_envelope, label='Envelope Inferior', color='red', linestyle='--')
        plt.fill_between(t*1e6, -amplitude_envelope, amplitude_envelope, color='red', alpha=0.1)
        plt.title('Sinal no Tempo com Envelope via Transformada de Hilbert')
        plt.xlabel('Tempo [μs]')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()'''

    except ImportError:
        messagebox.showerror("Erro", "A biblioteca PyEMD não está instalada. Use 'pip install EMD-signal' no terminal.")
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro na análise HHT: {str(e)}")

def find_tof():

    if signal is None or calibration_offset is None:
        messagebox.showwarning("Aviso", "Carregue um sinal!")
        return

    # ==================================================
    # PARÂMETROS
    # ==================================================

    dt_original = 0.5      # µs
    dt_interp = 0.01       # µs

    deriv_threshold = 1e-8

    tof_min_str = tof_ini.get()
    if not tof_min_str:
        messagebox.showwarning("Aviso", "Por favor, insira o valor de TOF estimado.")
        return
    
    tof_min = float(tof_min_str)

    central_freq_str = centralFreq.get()
    if not central_freq_str:
        messagebox.showwarning("Aviso", "Por favor, insira a frequência central.")
        return
        
    central_freq = float(central_freq_str) * 1e6 

    # ==================================================
    # SEM FILTRO (igual ao Pascal)
    # ==================================================

    signal_smooth = signal.copy()

    # ==================================================
    # INTERPOLAÇÃO TEMPORAL
    # ==================================================

    m = int(
        (len(signal_smooth) - 1)
        * (dt_original / dt_interp - 1)
        + len(signal_smooth)
    )

    tempo = np.zeros(m)

    for i in range(m):

        tempo[i] = (
            calibration_offset
            + dt_interp * i
        )

    # ==================================================
    # INTERPOLAÇÃO LINEAR
    # ==================================================

    amplitude = np.zeros(m)

    j = 0
    p = 0

    for i in range(m):

        if j >= len(signal_smooth) - 1:

            amplitude[i] = signal_smooth[-1]
            continue

        amplitude[i] = (
            (signal_smooth[j + 1]
            - signal_smooth[j])
            * (p / dt_original)
            + signal_smooth[j]
        )

        p += dt_interp

        if p > dt_original:

            p = 0
            j += 1

    # ==================================================
    # VALOR ABSOLUTO
    # ==================================================

    absoluto = np.abs(amplitude)

    # ==================================================
    # ENVELOPE CRESCENTE
    # ==================================================

    aux1 = np.zeros(m)

    aux1[0] = absoluto[0]

    for i in range(1, m):

        if absoluto[i] > aux1[i - 1]:

            aux1[i] = absoluto[i]

        else:

            aux1[i] = aux1[i - 1]

    # ==================================================
    # DERIVADA
    # ==================================================

    deriv = np.zeros(m)

    for i in range(m - 1):

        deriv[i] = (
            aux1[i + 1]
            - aux1[i]
        ) / dt_interp

    # ==================================================
    # DETECÇÃO DOS PICOS
    # ==================================================

    TD = []
    indices = []

    k = 0

    for i in range(m - 2):

        if (
            abs(deriv[i]) > deriv_threshold
            and k == 0
        ):

            for w in range(1, m - 2 - i):

                if abs(deriv[i + w]) > deriv_threshold:

                    k += 1

                else:

                    break

            if k > 0:

                TD.append(tempo[i])
                indices.append(i)

        elif abs(deriv[i]) <= deriv_threshold:

            k = 0

    # ==================================================
    # BUSCA DO TOF
    # ==================================================

    tzero = []

    for idx, td in zip(indices, TD):

        if td > tof_min:

            for i in range(idx, m - 1):

                amp1 = amplitude[i]
                amp2 = amplitude[i + 1]

                # cruzamento por zero
                if amp1 * amp2 < 0:

                    t = (
                        tempo[i + 1]
                        - amp2 * dt_interp
                        / (amp2 - amp1)
                    )

                    tzero.append(t)

                    break

                elif amp1 == 0 and amp2 != 0:

                    tzero.append(tempo[i])

                    break

                elif amp1 != 0 and amp2 == 0:

                    tzero.append(tempo[i + 1])

                    break

            break

    # ==================================================
    # RESULTADO
    # ==================================================

    if len(tzero) > 0:

        tof = tzero[0]+calibration_offset

        # ==================================================
        # PRIMEIRO PICO APÓS TOF
        # ==================================================

        # período da onda (µs)
        T_us = (1 / central_freq) * 1e6

        # janela de busca
        t_inicio = tof
        t_fim = tof + (T_us*1000)/2

        # índices da janela
        idx_window = np.where(
            (tempo >= t_inicio) &
            (tempo <= t_fim)
        )[0]

        # verificar se encontrou pontos
        if len(idx_window) > 0:

            # maior amplitude absoluta
            idx_local = np.argmax(
                np.abs(amplitude[idx_window])
            )

            idx_pico = idx_window[idx_local]

            amp_primeiro_pico = amplitude[idx_pico]

            tempo_pico = tempo[idx_pico]

        else:

            amp_primeiro_pico = None
            tempo_pico = None

        # textbox
        resultSPC_textbox.delete(1.0, "end")

        resultSPC_textbox.insert(
            "end",
            f"TOF: {tof:.4f} µs\n"
        )

        if amp_primeiro_pico is not None:
            resultSPC_textbox.insert(
                "end",
                f"Primeiro pico: {amp_primeiro_pico:.6f}\n"
            )

            resultSPC_textbox.insert(
                "end",
                f"Tempo do pico: {tempo_pico:.4f} µs\n"
            )

        # linha vertical
        line = ax1.axvline(
            x=tof,
            color='g',
            linestyle='--',
            linewidth=2,
            label='TOF'
        )

        elements_to_remove.append(line)

        ax1.legend()

        canvas.draw()

        return tof

    else:

        resultSPC_textbox.delete(1.0, "end")

        resultSPC_textbox.insert(
            "end",
            "TOF não encontrado\n"
        )

        canvas.draw()

        return None

def plot_wavelet_transform():
    if signal is None:
        messagebox.showerror("Erro", "Nenhum sinal carregado.")
        return

    # Tempo em milissegundos
    time = np.arange(len(signal)) / (acquisition_frequency * 1e3)

    # Wavelet e escalas
    wavelet = 'morl'
    scales = np.arange(7,50)

    # Transformada Wavelet
    coefficients, _ = pywt.cwt(signal, scales, wavelet)

    # Converter escalas para frequência (Hz)
    sampling_period = 1 / (acquisition_frequency * 1e3)  # em segundos
    frequencies = pywt.scale2frequency(wavelet, scales) / sampling_period
    frequencies /= 1e3

    # Plotar escalograma
    plt.figure(figsize=(10, 5))
    plt.imshow(np.abs(coefficients), extent=[time[0]*1e6, time[-1]*1e6, frequencies[-1], frequencies[0]],
               cmap='jet', aspect='auto')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Tempo (ms)')
    plt.ylabel('Frequência (kHz)')
    plt.title('Transformada Wavelet Contínua (Escalograma)')
    plt.tight_layout()
    plt.show()

def generate_sine_wave():
    # Novo formulário para inserir os parâmetros do sinal senoidal
    sine_window = ctk.CTkToplevel(app)
    sine_window.title("Generate sin. signal")

    # Garantir que a janela abra na frente
    sine_window.lift()
    sine_window.focus_force()
    # Esperar a janela ficar visível e então garantir que ela fique na frente
    sine_window.wait_visibility()
    sine_window.grab_set()  # Evita interação com a janela principal até que essa janela seja fechada
    sine_window.focus_force()

    ctk.CTkLabel(sine_window, text="Amplitude:").pack(pady=2)
    amplitude_entry = ctk.CTkEntry(sine_window, placeholder_text="1.0")
    amplitude_entry.pack(pady=2)

    ctk.CTkLabel(sine_window, text="Frequency (kHz):").pack(pady=2)
    frequency_entry = ctk.CTkEntry(sine_window, placeholder_text="1.0")
    frequency_entry.pack(pady=2)

    ctk.CTkLabel(sine_window, text="Time Step (µs):").pack(pady=2)
    timestep_entry = ctk.CTkEntry(sine_window, placeholder_text="1.0")
    timestep_entry.pack(pady=2)

    def generate_and_plot():
        try:
            amplitude = float(amplitude_entry.get())
            frequency = float(frequency_entry.get()) * 1000  # Convertendo kHz para Hz
            timestep = float(timestep_entry.get()) * 1e-6  # Convertendo µs para segundos

            # Calculando o número de pontos com base no passo de tempo e na largura do pulso
            t = np.arange(0, 0.0001, timestep)  # Usando intervalo de 1 segundo (pode ajustar)
            sine_signal = amplitude * np.sin(2 * np.pi * frequency * t)

            fig_sine, ax_sine = plt.subplots(figsize=(6, 4))
            ax_sine.plot(t * 1e6, sine_signal)  # Convertendo tempo para µs no gráfico
            ax_sine.set_title("Sin Signal")
            ax_sine.set_xlabel("Time (µs)")
            ax_sine.set_ylabel("Amplitude")
            ax_sine.grid(True)

            # Exibir o gráfico no novo formulário
            canvas_sine = FigureCanvasTkAgg(fig_sine, master=sine_window)
            canvas_sine.get_tk_widget().pack(pady=2)

            def export_signal():
                filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
                if filepath:
                    np.savetxt(filepath, sine_signal)
                    messagebox.showinfo("Success", f"Sin signal exported to {filepath}")

            export_button = ctk.CTkButton(sine_window, text="Export Signal", command=export_signal)
            export_button.pack(pady=2)

        except ValueError:
            messagebox.showerror("Erro", "Por favor, insira valores válidos.")

    ctk.CTkButton(sine_window, text="Generate and Plot", command=generate_and_plot).pack(pady=5)

def clear_elements():
    # Remove todos os elementos armazenados
    for element in elements_to_remove:
        element.remove()
    elements_to_remove.clear()
    canvas.draw()

def show_info_frame():
    # Criar a nova janela de informações
    info_window = ctk.CTkToplevel(app)
    info_window.title("About UltraNalyze")
    info_window.iconbitmap(resource_path("ultranalyze.ico"))
    info_window.geometry("700x500")
    
    # Carregar a imagem
    image_path = resource_path("ultranalyze.png")  # Substitua pelo caminho da sua imagem
    image = Image.open(image_path)
    image = image.resize((350, 150))  # Ajuste o tamanho da imagem conforme necessário
    photo = ImageTk.PhotoImage(image)
    
    # Adicionar a imagem
    image_label = ctk.CTkLabel(info_window, image=photo, text=None)
    image_label.image = photo  # Manter uma referência à imagem
    image_label.pack(pady=10)

    # Adicionar uma etiqueta com informações sobre o programa
    info_text = (
    "This software is designed for the nonlinear analysis of ultrasonic signals.\n"
    "It includes features for harmonic analysis, energy distribution assessment, and \n"
    "SPC index calculation. Developed as part of academic research at the São Carlos \n"
    "School of Engineering, University of São Paulo (USP), the program was created by \n"
    "PhD student Lara Guizi Anoni under the supervision of Prof. Dr. Vladimir Guilherme Haach. \n"
    " \n"
    "This program is for ACADEMIC USE ONLY. There is no guarantee of proper functioning \n"
    "of this software. The user is responsible for any and all conclusion made while using \n"
    "the program. Neither the author nor USP are legal responsible for any use or misuse of \n"
    "the program and the results. \n"
    )
    info_label = ctk.CTkLabel(info_window, text=info_text)
    info_label.pack(pady=20)

    # Botão para acessar o frame principal
    enter_button = ctk.CTkButton(info_window, text="Nonlinear signal analysis", fg_color="indianred", hover_color="darkred", command=lambda: [app.deiconify()])
    enter_button.pack(pady=10)

    # Botão para acessar a nova função
    new_function_button = ctk.CTkButton(info_window, text="Multiple signal analysis", fg_color="indianred", hover_color="darkred", command=multiple_files)
    new_function_button.pack(pady=10)

    # Adicionar o evento de fechamento
    info_window.protocol("WM_DELETE_WINDOW", app.destroy)

    # Inicialmente, esconder a janela principal
    app.withdraw()

# Função para criar a aba inicial
def create_tab(notebook, tab_name):
    global calibration_offset_entry, acquisition_frequency_entry, ax1, ax2, canvas
    global centralFreq, result_textbox, resultSPC_textbox, tof_ini
    global start_hanning_entry, end_hanning_entry, normalize_var
    global apply_hanning_var
    
    def hanning_geral():
        if apply_hanning_var.get():
            start_hanning_entry.configure(state="normal", text_color="black")
            end_hanning_entry.configure(state="normal", text_color="black")
        else:
            start_hanning_entry.configure(state="disabled", text_color="gray70")
            end_hanning_entry.configure(state="disabled", text_color="gray70")

    tab = ttk.Frame(notebook)
    notebook.add(tab, text=tab_name)

    normalize_var = ctk.BooleanVar(value=False)
    apply_hanning_var = ctk.BooleanVar(value=False)

    # Frame para o upload de dados
    upload_frame = ctk.CTkFrame(tab, width=300, height=500)  # Corrigido: widget dentro da aba
    upload_frame.grid(row=1, column=0, padx=2, pady=2, sticky="ns")

    ctk.CTkLabel(upload_frame, text="Upload Data").pack(pady=2)
    
    ctk.CTkLabel(upload_frame, text="Calibration Offset (sec):").pack(pady=2)
    calibration_offset_entry = ctk.CTkEntry(upload_frame, placeholder_text="-6.1")
    calibration_offset_entry.insert(0, "-6.1")
    calibration_offset_entry.pack(pady=2)

    ctk.CTkLabel(upload_frame, text="Acquisition Frequency (kHz):").pack(pady=2)
    acquisition_frequency_entry = ctk.CTkEntry(upload_frame, placeholder_text="2000.0")
    acquisition_frequency_entry.insert(0, "2000.0")
    acquisition_frequency_entry.pack(pady=2)

    # Checkbox: Apply Hanning Window
    apply_hanning_checkbox = ctk.CTkCheckBox(
        upload_frame,
        text="Apply Hanning Window",
        variable=apply_hanning_var,
        command=hanning_geral,
        checkbox_height=14,   # Altura do quadradinho
        checkbox_width=14,    # Largura do quadradinho
        border_width=2,       # Largura da borda do quadradinho
        fg_color="indianred",      # Cor de fundo do checkbox quando marcado
        hover_color="darkred",  # Cor ao passar o mouse
        text_color="black" )   # Cor do texto
    apply_hanning_checkbox.pack(pady=2)

    start_hanning_entry = ctk.CTkEntry(upload_frame, placeholder_text="50")
    end_hanning_entry = ctk.CTkEntry(upload_frame, placeholder_text="200")
    start_hanning_entry.insert(0, "20")
    start_hanning_entry.pack(pady=2)
    end_hanning_entry.insert(0, "500")
    end_hanning_entry.pack(pady=2)

    # Começa com os campos desabilitados
    start_hanning_entry.configure(state="disabled", text_color="gray70")
    end_hanning_entry.configure(state="disabled", text_color="gray70")

    normalize_checkbox = ctk.CTkCheckBox(
    upload_frame,
    text="Normalize Signal",
    variable=normalize_var,
    checkbox_height=14,   # Altura do quadradinho
    checkbox_width=14,    # Largura do quadradinho
    border_width=2,       # Largura da borda do quadradinho
    fg_color="indianred",      # Cor de fundo do checkbox quando marcado
    hover_color="darkred",  # Cor ao passar o mouse
    text_color="black" )   # Cor do texto
    normalize_checkbox.pack(pady=2)

    load_button = ctk.CTkButton(upload_frame, text="Load Signal", fg_color="indianred", hover_color="darkred", command=load_file)
    load_button.pack(pady=2)

    # Frame para Análise Harmônica
    harmonic_frame = ctk.CTkFrame(tab, width=300, height=500)  # Corrigido
    harmonic_frame.grid(row=1, column=1, padx=2, pady=2, sticky="ns")

    ctk.CTkLabel(harmonic_frame, text="Harmonic Analysis").pack(pady=2)

    ctk.CTkLabel(harmonic_frame, text="Central Emission Frequency (kHz):").pack(pady=2)
    centralFreq = ctk.CTkEntry(harmonic_frame, placeholder_text="54")
    centralFreq.insert(0, "54")
    centralFreq.pack(pady=2)

    # Frame para os botões organizados em uma linha
    button_frame = ctk.CTkFrame(harmonic_frame)
    button_frame.pack(pady=10)
    # Botão para encontrar harmônicos
    find_button = ctk.CTkButton(button_frame, text="Find Harmonics", fg_color="indianred", hover_color="darkred", command=find_harmonics)
    find_button.grid(row=0, column=0, padx=(0, 10))  # Espaço à direita
    # Botão para plotar harmônicos
    ED_button = ctk.CTkButton(button_frame, text="Energy Distribution", fg_color="indianred", hover_color="darkred", command=calculate_energy_distribution)
    ED_button.grid(row=0, column=1)

    result_textbox = ctk.CTkTextbox(harmonic_frame, width=250, height=100)
    result_textbox.pack(pady=2)

    # Frame para os botões organizados em uma linha
    button_frame2 = ctk.CTkFrame(harmonic_frame)
    button_frame2.pack(pady=10)
    # Botão para plotar
    plot_button = ctk.CTkButton(button_frame2, text="Plot Harmonics", fg_color="indianred", hover_color="darkred", command=plot_harmonics)
    plot_button.grid(row=0, column=0, padx=(0, 10)) 
    # Botão para apagar
    clear_button = ctk.CTkButton(button_frame2, text="Clear FFT graph", fg_color="indianred", hover_color="darkred", command=clear_elements)
    clear_button.grid(row=0, column=1)

    # Frame para SPC e HHT
    SPC_frame = ctk.CTkFrame(tab, width=300, height=500)
    SPC_frame.grid(row=1, column=2, padx=2, pady=2, sticky="ns")
    ctk.CTkLabel(SPC_frame, text="Secundary Analysis").pack(pady=2)

    # Botão SPC
    SPC_button = ctk.CTkButton(SPC_frame, text="SPC-I", fg_color="indianred", hover_color="darkred", command=SPC)
    SPC_button.pack(pady=2)

    # Botão HHT
    HHT_button = ctk.CTkButton(SPC_frame, text="Apply HHT", fg_color="indianred", hover_color="darkred", command=HHT)
    HHT_button.pack(pady=2)

    # Botão TOF
    TOF_bt = ctk.CTkButton(SPC_frame, text="Find TOF", fg_color="indianred", hover_color="darkred", command=find_tof)
    TOF_bt.pack(pady=2)
    ctk.CTkLabel(SPC_frame, text="Estimated TOF (µs):").pack(pady=2)
    tof_ini = ctk.CTkEntry(SPC_frame, placeholder_text="25")
    tof_ini.insert(0, "25")
    tof_ini.pack(pady=2)

    # Caixa de resultado do SPC
    resultSPC_textbox = ctk.CTkTextbox(SPC_frame, width=250, height=100)
    resultSPC_textbox.pack(pady=2)


    # Frame de outras funcionalidades
    other_frame = ctk.CTkFrame(tab, width=300, height=500)
    other_frame.grid(row=1, column=3, padx=2, pady=2, sticky="ns")

    ctk.CTkLabel(other_frame, text="Other functionalities").pack(pady=2)
    generate_sine_button = ctk.CTkButton(other_frame, text="Generate sin. signal", fg_color="indianred", hover_color="darkred", command=generate_sine_wave)
    generate_sine_button.pack(pady=2)

    org_bt = ctk.CTkButton(other_frame, text="Proccess signals (raw data)", fg_color="indianred", hover_color="darkred", command=process_RAWsignals)
    org_bt.pack(pady=2)

    wt_bt = ctk.CTkButton(other_frame, text="Wavelet Transform", fg_color="indianred", hover_color="darkred", command=plot_wavelet_transform)
    wt_bt.pack(pady=2)

    tab_button_new = ctk.CTkButton(other_frame, text="New Tab", fg_color="indianred", hover_color="darkred", command=add_tab)
    tab_button_new.pack(pady=2)

    tab_del_bt = ctk.CTkButton(other_frame, text="Delete Tab", fg_color="indianred", hover_color="darkred", command=delete_current_tab)
    tab_del_bt.pack(pady=2)

    # Configurar área para os gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
    fig.tight_layout(pad=3.0)

    canvas = FigureCanvasTkAgg(fig, master=tab)
    canvas.get_tk_widget().grid(row=2, column=0, columnspan=4, padx=2, pady=2, sticky="nsew")

    toolbar = NavigationToolbar2Tk(canvas, tab, pack_toolbar=False)
    toolbar.update()
    toolbar.grid(row=3, column=0, columnspan=4, padx=2, pady=2, sticky="nsew")

    # Configurar redimensionamento das colunas e linhas
    tab.grid_columnconfigure(0, weight=1)
    tab.grid_columnconfigure(1, weight=1)
    tab.grid_columnconfigure(2, weight=1)
    tab.grid_columnconfigure(3, weight=1)
    tab.grid_rowconfigure(0, weight=1)
    tab.grid_rowconfigure(1, weight=1)
    tab.grid_rowconfigure(2, weight=1)
    tab.grid_rowconfigure(3, weight=0)

# Função para adicionar uma nova aba
def add_tab():
    tab_name = f"Tab {notebook.index('end') + 1}"
    create_tab(notebook, tab_name)

# Funçao para deletar aba
def delete_current_tab():
    # Verifica se há mais de uma aba, para não deletar todas
    if notebook.index('end') > 1:
        current_tab = notebook.select()  # Obtém o identificador da aba atual
        notebook.forget(current_tab)     # Deleta a aba atual
    else:
        messagebox.showwarning("Atenção", "Você não pode deletar a única aba restante.")

# Vários sinais
def multiple_files():
    global normalize_var_m
    # Novo formulário para inserir os parâmetros do sinal senoidal
    multi_window = ctk.CTkToplevel(app)
    multi_window.title("Multiple files data analysis")

    normalize_var_m = ctk.BooleanVar(value=False)

    # Garantir que a janela abra na frente
    multi_window.lift()
    multi_window.focus_force()
    multi_window.wait_visibility()
    multi_window.grab_set()
    multi_window.focus_force()

    ctk.CTkLabel(multi_window, text="Parameters config.").pack(pady=2)
    ctk.CTkLabel(multi_window, text="Calibration Offset (sec):").pack(pady=2)
    calibration = ctk.CTkEntry(multi_window, placeholder_text="-6.1")
    calibration.insert(0, "-6.1")
    calibration.pack(pady=2)

    ctk.CTkLabel(multi_window, text="Acquisition Frequency (kHz):").pack(pady=2)
    acquisition = ctk.CTkEntry(multi_window, placeholder_text="2000.0")
    acquisition.insert(0, "2000.0")
    acquisition.pack(pady=2)

    ctk.CTkLabel(multi_window, text="Central Emission Frequency (kHz):").pack(pady=2)
    centralF = ctk.CTkEntry(multi_window, placeholder_text="54")
    centralF.insert(0, "54")
    centralF.pack(pady=2)

    ctk.CTkLabel(multi_window, text="Estimated TOF (µs):").pack(pady=2)
    tof_est = ctk.CTkEntry(multi_window, placeholder_text="25")
    tof_est.insert(0, "25")
    tof_est.pack(pady=2)

    normalize_checkbox_m = ctk.CTkCheckBox(
    multi_window,
    text="Normalize Signal",
    variable=normalize_var_m,
    checkbox_height=14,   # Altura do quadradinho
    checkbox_width=14,    # Largura do quadradinho
    border_width=2,       # Largura da borda do quadradinho
    fg_color="indianred",      # Cor de fundo do checkbox quando marcado
    hover_color="darkred",  # Cor ao passar o mouse
    text_color="black" )   # Cor do texto
    normalize_checkbox_m.pack(pady=2)

    def calculate_energy_distribution_m(start, end):
        global found_harmonics, signal, acquisition_frequency
        if signal is None or acquisition_frequency is None:
            messagebox.showwarning("Aviso", "Carregue um sinal e defina a frequência de aquisição antes de encontrar harmônicos.")
            return
        
        try:
            central_freq_str = float(centralF.get())
            if not central_freq_str:
                messagebox.showwarning("Aviso", "Por favor, insira a frequência central.")
                return
            
            central_freq = float(central_freq_str) * 1e6

            N = len(signal)
            T = 1.0 / (acquisition_frequency * 1e6)

            if start and end != 0:
                # Definir o intervalo Hanning Window
                window_length = end - start
                window = np.hanning(window_length)
                # Criar um sinal ajustado onde o intervalo fora da janela é zerado
                windowed_signal = np.zeros_like(signal)  # Sinal zerado
                windowed_signal[start:end] = signal[start:end] * window
                # Calcular a FFT do sinal com janela parcial
                yf = fft(windowed_signal)
                xf = fftfreq(N, T)[:N//2]
            else:
                # Calcular a FFT do sinal
                yf = fft(signal)
                xf = fftfreq(N, T)[:N//2]

            
            # Encontrar o pico máximo
            max_amp = np.max(np.abs((2.0/N) * np.abs(yf[:N//2])))

            # Encontrar picos na FFT
            peaks, _ = find_peaks((2.0/N) * np.abs(yf[:N//2]), height=0.05*max_amp) 
            
            # Encontrar pico máximo ao redor da frequência central
            tolerance = 0.3 * central_freq  # Tolerância de 20% ao redor da frequência central
            max_amplitude = 0
            central_peak = None

            for peak in peaks:
                if abs(xf[peak] - central_freq) <= tolerance:
                    amplitude = (2.0/N) * np.abs(yf[peak])
                    if amplitude > max_amplitude:
                        max_amplitude = amplitude
                        central_peak = peak
            if central_peak is None:
                messagebox.showwarning("Aviso", "Não foi encontrado um pico significativo ao redor da frequência central.")
                energy_distribution=0
                return
        
            # Calcular a energia na banda central (central_freq ± 5 kHz)
            central_band = (xf >= (xf[central_peak] - 5* 1e6)) & (xf <= (xf[central_peak] + 5* 1e6))
            yf_central = yf[:N//2]
            central_energy = np.sum(((2.0/N)*np.abs(yf_central[central_band]))**2)
            
            # Calcular a energia nas bandas laterais (50 kHz de cada lado)
            left_band = (xf >= (xf[central_peak] - 55* 1e6)) & (xf < (xf[central_peak] - 5* 1e6))
            right_band = (xf > (xf[central_peak] + 5* 1e6)) & (xf <= (xf[central_peak] + 55* 1e6))
            left_energy = np.sum(((2.0/N)*np.abs(yf_central[left_band]))**2)
            right_energy = np.sum(((2.0/N)*np.abs(yf_central[right_band]))**2)
            
            # Calcular a distribuição de energia
            if central_energy != 0:
                energy_distribution = (left_energy + right_energy)/central_energy
            else:
                energy_distribution=0
                messagebox.showwarning("Aviso", "Energia zero encontrada nas bandas central, não é possível calcular a distribuição de energia.")
            
            return energy_distribution

        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro no cálculo da distribuição de energia: {str(e)}")
            return 0
        
    def find_harmonics_m(start, end, H_folder):
        global found_harmonics, signal, acquisition_frequency
        if signal is None or acquisition_frequency is None:
            messagebox.showwarning("Aviso", "Carregue um sinal e defina a frequência de aquisição antes de encontrar harmônicos.")
            return
        
        try:
            central_freq_str = float(centralF.get())
            if not central_freq_str:
                messagebox.showwarning("Aviso", "Por favor, insira a frequência central.")
                return
            
            central_freq = float(central_freq_str) * 1e6
            central_freq_real = central_freq

            N = len(signal)
            T = 1.0 / (acquisition_frequency * 1e6)

            if start != 0 and end != 0:
                # Definir o intervalo Hanning Window
                window_length = end - start
                window = np.hanning(window_length)
                # Criar um sinal ajustado onde o intervalo fora da janela é zerado
                windowed_signal = np.zeros_like(signal)  # Sinal zerado
                windowed_signal[start:end] = signal[start:end] * window
                # Calcular a FFT do sinal com janela parcial
                yf = fft(windowed_signal)
                xf = fftfreq(N, T)[:N//2]
            else:
                # Calcular a FFT do sinal
                yf = fft(signal)
                xf = fftfreq(N, T)[:N//2] 

            harmonic_frequencies = []
            harmonic_amplitudes = []

            # Procurar picos ao redor da frequência central e seus harmônicos
            for i in range(1, 5):  # Consideramos os primeiros 4 harmônicos
                if i == 1:
                    target_freq = i * central_freq
                    tolerance = 0.2 * central_freq  # Tolerância de 20% ao redor da frequência
                else:
                    target_freq = i * central_freq_real
                    tolerance = 0.3 * central_freq_real

                # Encontrar os índices dentro da faixa desejada
                valid_indices = np.where((xf >= target_freq - tolerance) & (xf <= target_freq + tolerance))[0]

                if valid_indices.size > 0:  # Se existir pelo menos um valor dentro do intervalo
                    best_index = max(valid_indices, key=lambda k: np.abs(yf[k]))  # Maior amplitude dentro da faixa
                    harmonic_freq = xf[best_index]
                    harmonic_amp = (2.0 / N) * np.abs(yf[best_index])

                    harmonic_frequencies.append(harmonic_freq)
                    harmonic_amplitudes.append(harmonic_amp)

                    if i == 1:
                        central_freq_real = harmonic_freq  # Atualizar a frequência real do primeiro harmônico

            # Escrever os resultados no arquivo de saída
            result_values = []
            for i, freq in enumerate(harmonic_frequencies):
                amp=harmonic_amplitudes[i]
                if isinstance(freq, (int, float)):  # Se for um número, converter para MHz
                    freq_c=freq/1e6
                    result_values.append(freq_c)
                    result_values.append(amp)
                else:  # Se for "x", apenas escrever como string
                    result_values.append(freq)
                    result_values.append(amp)
            
            # Calcular e escrever o valor de beta
            if len(harmonic_amplitudes) > 1:  # Verificar se há ao menos 2 harmônicos para calcular beta
                A1 = harmonic_amplitudes[0]  # Amplitude do primeiro harmônico (fundamental)
                A2 = harmonic_amplitudes[1]  # Amplitude do segundo harmônico (índice 2)
                A3 = harmonic_amplitudes[2]

                if A1 != 0:
                    beta = A2 / (A1 ** 2)
                    gama = A3 / (A1 ** 3)
                    result_values.append(beta)
                    result_values.append(gama)
                else:
                    result_values.append(float('nan'))
            else:
                result_values.append(0)

            # Salvar as frequências e amplitudes para visualização futura, se necessário
            found_harmonics = (harmonic_frequencies, harmonic_amplitudes)

            # --------- SALVAR GRÁFICO DA FFT COM HARMÔNICOS ---------
            plt.figure(figsize=(10, 5))
            plt.plot(xf / 1e6, (2.0/N) * np.abs(yf[:N//2]), label='FFT')
            for i, freq in enumerate(harmonic_frequencies):
                plt.axvline(x=freq / 1e6, color='red', linestyle='--', label=f'H{i+1}' if i == 0 else None)

            plt.xlabel("Frequência (kHz)")
            plt.ylabel("Amplitude")
            plt.title("FFT com Harmônicos Detectados")
            plt.grid(True)
            plt.legend()

            # Nome do arquivo com timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"fft_harmonics_{timestamp}.png"
            save_path = os.path.join(H_folder, filename)

            plt.savefig(save_path, dpi=300)
            plt.close()
            
            return result_values
        
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro na análise harmônica: {str(e)}")

    def SPC_m(start, end):
        global signal, acquisition_frequency
        if signal is None or acquisition_frequency is None:
            messagebox.showwarning("Aviso", "Carregue um sinal e defina a frequência de aquisição antes de realizar a análise SPC.")
            return

        try:

            N = len(signal)
            T = 1.0 / (acquisition_frequency * 1e6)

            if start and end != 0:
                # Definir o intervalo Hanning Window
                window_length = end - start
                window = np.hanning(window_length)
                # Criar um sinal ajustado onde o intervalo fora da janela é zerado
                windowed_signal = np.zeros_like(signal)  # Sinal zerado
                windowed_signal[start:end] = signal[start:end] * window
                # Calcular a FFT do sinal com janela parcial
                yf = fft(windowed_signal)
                xf = fftfreq(N, T)[:N//2]
            else:
                # Calcular a FFT do sinal
                yf = fft(signal)
                xf = fftfreq(N, T)[:N//2]

            # Encontrar o pico máximo
            max_amplitude = np.max(np.abs((2.0/N) * np.abs(yf[:N//2])))

            # Encontrar picos na FFT
            peaks, _ = find_peaks((2.0/N) * np.abs(yf[:N//2]), height=0.05*max_amplitude)

            # Somar amplitudes dos picos
            peak_amplitudes = (2.0/N) * np.abs(yf[peaks])
            total_amplitude = np.sum(peak_amplitudes)

            # Calcular média das amplitudes
            spi = total_amplitude / max_amplitude
            return [spi]

        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro na análise SPC: {str(e)}")      

    def HHT_m(start, end):
        global signal, acquisition_frequency

        if signal is None or acquisition_frequency is None:
            messagebox.showwarning("Aviso", "Carregue um sinal e defina a frequência de aquisição antes de aplicar a HHT.")
            return

        try:
            N = len(signal)
            T = 1.0 / (acquisition_frequency * 1e6)
            t = np.linspace(0, N*T, N)

            if start and end != 0:
                # Definir o intervalo Hanning Window
                window_length = end - start
                window = np.hanning(window_length)
                # Criar um sinal ajustado onde o intervalo fora da janela é zerado
                windowed_signal = np.zeros_like(signal)  # Sinal zerado
                windowed_signal[start:end] = signal[start:end] * window
                signal_to_analyze = windowed_signal
            else:
                # Calcular a FFT do sinal
                signal_to_analyze = signal

            result_values = []
            # Aplicar Hilbert no sinal diretamente
            analytic_signal = hilbert(signal_to_analyze)
            amplitude_envelope = np.abs(analytic_signal)

            # Cálculo da área sob o envelope
            area_envelope = trapz(amplitude_envelope, t)
            result_values.append(area_envelope*1e6)

            #Para a frequnecia
            yf = fft(signal_to_analyze)
            xf = fftfreq(N, T)[:N // 2] 
            magnitude_spectrum = 2.0 / N * np.abs(yf[:N // 2])  # módulo da FFT = envelope espectral
            area_spectral_envelope = trapz(magnitude_spectrum, xf)
            result_values.append(area_spectral_envelope*1e-3)

            return result_values
        
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro na análise HHT: {str(e)}")

    def find_tof():
        global signal, acquisition_frequency, tof_estimated, calibration_offset, central_frequency
        if signal is None or acquisition_frequency is None:
            messagebox.showwarning("Aviso", "Carregue um sinal e defina a frequência de aquisição antes de realizar a análise SPC.")
            return

        # ==================================================
        # PARÂMETROS
        # ==================================================
        dt_original = 0.5      # µs
        dt_interp = 0.01       # µs
        deriv_threshold = 1e-8
        tof_min = tof_estimated
        central_freq = central_frequency*1e6 

        # ==================================================
        # SEM FILTRO (igual ao Pascal)
        # ==================================================
        signal_smooth = signal.copy()

        # ==================================================
        # INTERPOLAÇÃO TEMPORAL
        # ==================================================
        m = int(
            (len(signal_smooth) - 1)
            * (dt_original / dt_interp - 1)
            + len(signal_smooth)
        )

        tempo = np.zeros(m)
        for i in range(m):
            tempo[i] = (
                calibration_offset
                + dt_interp * i
            )

        # ==================================================
        # INTERPOLAÇÃO LINEAR
        # ==================================================
        amplitude = np.zeros(m)
        j = 0
        p = 0

        for i in range(m):
            if j >= len(signal_smooth) - 1:

                amplitude[i] = signal_smooth[-1]
                continue
            amplitude[i] = (
                (signal_smooth[j + 1]
                - signal_smooth[j])
                * (p / dt_original)
                + signal_smooth[j]
            )

            p += dt_interp
            if p > dt_original:
                p = 0
                j += 1

        # ==================================================
        # VALOR ABSOLUTO
        # ==================================================
        absoluto = np.abs(amplitude)

        # ==================================================
        # ENVELOPE CRESCENTE
        # ==================================================
        aux1 = np.zeros(m)
        aux1[0] = absoluto[0]

        for i in range(1, m):
            if absoluto[i] > aux1[i - 1]:
                aux1[i] = absoluto[i]
            else:
                aux1[i] = aux1[i - 1]

        # ==================================================
        # DERIVADA
        # ==================================================

        deriv = np.zeros(m)

        for i in range(m - 1):
            deriv[i] = (
                aux1[i + 1]
                - aux1[i]
            ) / dt_interp

        # ==================================================
        # DETECÇÃO DOS PICOS
        # ==================================================
        TD = []
        indices = []
        k = 0

        for i in range(m - 2):
            if (
                abs(deriv[i]) > deriv_threshold
                and k == 0
            ):
                for w in range(1, m - 2 - i):

                    if abs(deriv[i + w]) > deriv_threshold:
                        k += 1
                    else:
                        break

                if k > 0:
                    TD.append(tempo[i])
                    indices.append(i)

            elif abs(deriv[i]) <= deriv_threshold:
                k = 0

        # ==================================================
        # BUSCA DO TOF
        # ==================================================
        tzero = []
        for idx, td in zip(indices, TD):
            if td > tof_min:
                for i in range(idx, m - 1):
                    amp1 = amplitude[i]
                    amp2 = amplitude[i + 1]

                    # cruzamento por zero
                    if amp1 * amp2 < 0:

                        t = (
                            tempo[i + 1]
                            - amp2 * dt_interp
                            / (amp2 - amp1)
                        )

                        tzero.append(t)
                        break

                    elif amp1 == 0 and amp2 != 0:
                        tzero.append(tempo[i])
                        break

                    elif amp1 != 0 and amp2 == 0:
                        tzero.append(tempo[i + 1])
                        break
                break

        # ==================================================
        # RESULTADO
        # ==================================================
        if len(tzero) > 0:
            result_values = []
            tof = tzero[0]+calibration_offset
            result_values.append(tof)

            # ==================================================
            # PRIMEIRO PICO APÓS TOF
            # ==================================================

            # período da onda (µs)
            T_us = (1 / central_freq) * 1e6

            # janela de busca
            t_inicio = tof
            t_fim = tof + (T_us*1000)/2

            # índices da janela
            idx_window = np.where(
                (tempo >= t_inicio) &
                (tempo <= t_fim)
            )[0]

            # verificar se encontrou pontos
            if len(idx_window) > 0:

                # maior amplitude absoluta
                idx_local = np.argmax(
                    np.abs(amplitude[idx_window])
                )

                idx_pico = idx_window[idx_local]

                amp_primeiro_pico = amplitude[idx_pico]
                tempo_pico = tempo[idx_pico]

            else:

                amp_primeiro_pico = None
                tempo_pico = None

            result_values.append(amp_primeiro_pico)
            result_values.append(tempo_pico)

            return result_values

    def load_multiple_files():
        global signal, acquisition_frequency, normalize_var_m, calibration_offset, tof_estimated, central_frequency
        filepaths = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt")])

        normalize_m = normalize_var_m.get()

        if not filepaths:
            messagebox.showwarning("Attention", "No file selected.")
            return

        apply_hanning = messagebox.askyesno("Apply Hanning Window", "Do you want to apply a Hanning window before FFT?")
        start, end = None, None

        if apply_hanning:
            start = simpledialog.askinteger("Start", "Enter the start index:", parent=multi_window)
            end = simpledialog.askinteger("End", "Enter the end index:", parent=multi_window)   
        else:
            start, end = 0, 0
        
        try:
            calibration_offset = float(calibration.get())
            acquisition_frequency = float(acquisition.get())
            tof_estimated = float(tof_est.get())
            central_frequency = float(centralF.get())

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            full_analysis_name = f"full_analysis_{timestamp}.csv"
            error_report_name = f"error_report_{timestamp}.csv"

            with open(full_analysis_name, "w") as output_file, \
                open(error_report_name, "w") as error_file:

                
                output_file.write(f"Fre1; H1; Freq2; H2; Freq3; H3; Freq4; H4; Beta; Gama; SPI; Energy Distribution; Envelope; FFT envelope; TOF; 1st peak; Time 1st peak\n")

                # Criar pasta de saída para salvar as imagens
                H_folder = "FFT_Harmonics_Images"
                os.makedirs(H_folder, exist_ok=True)

                for filepath in filepaths:
                    signal = np.loadtxt(filepath)
                    if normalize_m:
                        signal = signal / np.max(np.abs(signal))  # Normalização simples [-1, 1]
                    
                    time = (np.arange(len(signal)) / (acquisition_frequency * 1e6))  # Convertendo kHz para Hz
                    time += calibration_offset*1e-6

                    if apply_hanning:
                        # Verificar se algum valor do sinal no intervalo atingiu 100
                        if np.any(signal[start:end] >= 99):
                            error_file.write(f"Verificar {filepath} \n")   
                    else:
                        if np.any(signal >= 99):
                            error_file.write(f"Verificar {filepath} \n")
                    
                    harmonics_data=find_harmonics_m(start, end, H_folder)
                    spi_data=SPC_m(start, end)
                    ed_data=calculate_energy_distribution_m(start, end)
                    ed_data=[ed_data]
                    hht_data=HHT_m(start, end)
                    tof_data=find_tof()

                    combined_data = harmonics_data + spi_data + ed_data + hht_data + tof_data
                    output_file.write(";".join(map(str, combined_data)) + "\n")

        
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro ao carregar os arquivos: {str(e)}")
    
        messagebox.showinfo("Success", "Processing completed! Results saved in harmonics_analysis.csv, SPC_analysis.csv, and error_report.csv")
        print("Processing completed!")

    ctk.CTkButton(multi_window, text="Load Multiple Files", fg_color="indianred", hover_color="darkred", command=load_multiple_files).pack(pady=5)

# Configuração da janela principal
app = ctk.CTk()
app.geometry("1000x700")
app.title("Nonlinear ultrasound analysis")
app.iconbitmap(resource_path("ultranalyze.ico"))

# Criar o widget Notebook para gerenciar abas
notebook = ttk.Notebook(app)
notebook.grid(row=0, column=0, columnspan=4, padx=2, pady=2, sticky="nsew")

# Criar a aba inicial
create_tab(notebook, "Initial Tab")

# Configurar redimensionamento das colunas e linhas
app.grid_columnconfigure(0, weight=1)
app.grid_columnconfigure(1, weight=1)
app.grid_columnconfigure(2, weight=1)
app.grid_columnconfigure(3, weight=1)
app.grid_rowconfigure(0, weight=1)
app.grid_rowconfigure(1, weight=1)
app.grid_rowconfigure(2, weight=1)
app.grid_rowconfigure(3, weight=0)

# Exibir o frame de informações
show_info_frame()

# Executar a aplicação
app.mainloop()