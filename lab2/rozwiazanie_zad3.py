import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")

# 1. Wczytanie danych z arkusza 'Dane'
df = pd.read_excel('dane.xlsx', sheet_name='Dane')
df.columns = df.columns.astype(str).str.strip()

# 2. Filtracja i wybór cechy "Skocz" dla dwunastolatków
df_12 = df[(df['Wiek'] >= 12.00) & (df['Wiek'] < 13.00)].copy()
# Odrzucenie braków danych
skocz_data = df_12['Skocz'].dropna().values
n = len(skocz_data)

# --- STATYSTYKI DLA PRÓBY NIEUPORZĄDKOWANEJ ---
mean_raw = np.mean(skocz_data)
std_raw = np.std(skocz_data, ddof=1)
median_raw = np.median(skocz_data)
cv_raw = (std_raw / mean_raw) * 100

err_mean_raw = std_raw / np.sqrt(n)
err_std_raw = std_raw / np.sqrt(2*n)
err_median_raw = 1.2533 * err_mean_raw

mode_raw_res = stats.mode(np.round(skocz_data, 2), keepdims=True)
mode_raw = mode_raw_res.mode[0]
err_mode_raw = np.nan

skew_raw = stats.skew(skocz_data, bias=False)
mixed_skew_raw = 3 * (mean_raw - median_raw) / std_raw

t1_l, t1_h = mean_raw - std_raw, mean_raw + std_raw
t2_l, t2_h = mean_raw - 2*std_raw, mean_raw + 2*std_raw
pct_t1_raw = np.mean((skocz_data >= t1_l) & (skocz_data <= t1_h)) * 100
pct_t2_raw = np.mean((skocz_data >= t2_l) & (skocz_data <= t2_h)) * 100
out_right_raw = np.sum(skocz_data > t2_h)
out_left_raw = np.sum(skocz_data < t2_l)

# --- FUNKCJA DO STATYSTYK Z SZEREGÓW ROZDZIELCZYCH ---
def calc_grouped_stats(bins_edges, data):
    frequencies, edges = np.histogram(data, bins=bins_edges)
    h_arr = np.diff(edges)
    midpoints = edges[:-1] + h_arr/2
    n_g = np.sum(frequencies)
    
    if n_g == 0:
        return [np.nan]*15
        
    mean_g = np.sum(midpoints * frequencies) / n_g
    var_g = np.sum(frequencies * (midpoints - mean_g)**2) / (n_g - 1)
    std_g = np.sqrt(var_g)
    
    d_idx = np.argmax(frequencies)
    L_d = edges[d_idx]
    f_d = frequencies[d_idx]
    f_d_minus_1 = frequencies[d_idx-1] if d_idx > 0 else 0
    f_d_plus_1 = frequencies[d_idx+1] if d_idx < len(frequencies)-1 else 0
    h_d = h_arr[d_idx]
    
    denom = (f_d - f_d_minus_1) + (f_d - f_d_plus_1)
    mode_g = L_d + ((f_d - f_d_minus_1) / denom) * h_d if denom != 0 else L_d + h_d/2
    
    cum_freq = np.cumsum(frequencies)
    m_idx = np.where(cum_freq >= n_g/2)[0][0]
    L_m = edges[m_idx]
    F_m_minus_1 = cum_freq[m_idx-1] if m_idx > 0 else 0
    f_m = frequencies[m_idx]
    h_m = h_arr[m_idx]
    median_g = L_m + ((n_g/2 - F_m_minus_1) / f_m) * h_m if f_m != 0 else L_m + h_m/2
    
    err_mean_g = std_g / np.sqrt(n_g)
    err_std_g = std_g / np.sqrt(2*n_g)
    err_median_g = 1.2533 * err_mean_g
    err_mode_g = np.nan 
    
    cv_g = (std_g / mean_g) * 100
    skew_g_mixed = 3 * (mean_g - median_g) / std_g if std_g != 0 else np.nan
    skew_g_class = (mean_g - mode_g) / std_g if std_g != 0 else np.nan
    
    return [
        round(mean_g, 2), round(err_mean_g, 3), round(std_g, 2), round(err_std_g, 3), 
        round(median_g, 2), round(err_median_g, 3), round(mode_g, 2), err_mode_g, 
        round(cv_g, 2), round(skew_g_class, 2), round(skew_g_mixed, 2), 
        'x', 'x', 'x', 'x' # Wartości typowe/odstające w szeregu zostawiamy z 'x'
    ]

# --- PARAMETRY SZEREGU "NAJLEPSZEGO" ---
k_sturges = int(np.ceil(1 + 3.322 * np.log10(n)))
r = np.max(skocz_data) - np.min(skocz_data)
h_best = r / k_sturges
# Punkt początkowy przesuwamy lekko w dół, dla danych całkowitych np. -0.5
x01_best = np.min(skocz_data) - 0.5 
bins_best = [x01_best + i*h_best for i in range(k_sturges + 1)]
bins_best[-1] = max(bins_best[-1], np.max(skocz_data) + 0.1)

# --- ZBIERANIE WYNIKÓW ---
results = {
    'próba nieuporządkowana': [
        round(mean_raw, 2), round(err_mean_raw, 3), round(std_raw, 2), round(err_std_raw, 3), 
        round(median_raw, 2), round(err_median_raw, 3), round(mode_raw, 2), np.nan, 
        round(cv_raw, 2), round(skew_raw, 2), round(mixed_skew_raw, 2), 
        round(pct_t1_raw, 2), round(pct_t2_raw, 2), out_right_raw, out_left_raw
    ],
    'szereg "najlepszy"': calc_grouped_stats(bins_best, skocz_data)
}

index_names = [
    'średnia', 'błąd średniej', 'odch. Standard.', 'bład odchylenia', 
    'mediana', 'bład mediany', 'dominanta', 'bład dominanty', 
    'wsp. zmienności', 'wsp. asymetrii', 'mieszany wsp.asymetrii', 
    'procent typowe1', 'procent typowe2', 
    'liczba wartości odstających po prawej stronie', 'liczba wartości odstających po lewej stronie'
]

df_res = pd.DataFrame(results, index=index_names)
df_res.to_excel('wyniki_zad3.xlsx')
print("Sukces! Wygenerowano plik 'wyniki_zad3.xlsx'")