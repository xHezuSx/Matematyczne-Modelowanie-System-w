import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore") # Ukrywa ostrzeżenia o dzieleniu przez zero itp.

# 1. Wczytanie danych z właściwego arkusza ('Dane')
df = pd.read_excel('dane.xlsx', sheet_name='Dane')

# Usunięcie ewentualnych spacji w nazwach kolumn
df.columns = df.columns.astype(str).str.strip()

# 2. Filtracja dwunastolatków (wiek 12.00-12.99)
df_12 = df[(df['Wiek'] >= 12.00) & (df['Wiek'] < 13.00)].copy()

# 3. Obliczenie współczynnika Rohrera
# Wzrost z cm na m
df_12['Wzrost_m'] = df_12['Wys'] / 100
# Wzór na wsp. Rohrera: Masa [kg] / (Wzrost [m])^3
df_12['Rohrer'] = df_12['Masa'] / (df_12['Wzrost_m']**3)

# Odrzucenie ewentualnych braków danych
rohrer_data = df_12['Rohrer'].dropna().values
n = len(rohrer_data)

# --- STATYSTYKI DLA PRÓBY NIEUPORZĄDKOWANEJ ---
mean_raw = np.mean(rohrer_data)
std_raw = np.std(rohrer_data, ddof=1)
median_raw = np.median(rohrer_data)
cv_raw = (std_raw / mean_raw) * 100

# Błędy statystyczne
err_mean_raw = std_raw / np.sqrt(n)
err_std_raw = std_raw / np.sqrt(2*n)
err_median_raw = 1.2533 * err_mean_raw

# Dominanta z próby (używamy mody z zaokrąglonych wartości dla sensownego wyniku ciągłego)
mode_raw_res = stats.mode(np.round(rohrer_data, 2), keepdims=True)
mode_raw = mode_raw_res.mode[0]
err_mode_raw = np.nan # Błąd dominanty dla próby nieuporządkowanej zazwyczaj się pomija

# Asymetria
skew_raw = stats.skew(rohrer_data, bias=False)
mixed_skew_raw = 3 * (mean_raw - median_raw) / std_raw

# Wartości typowe i odstające
t1_l, t1_h = mean_raw - std_raw, mean_raw + std_raw
t2_l, t2_h = mean_raw - 2*std_raw, mean_raw + 2*std_raw
pct_t1_raw = np.mean((rohrer_data >= t1_l) & (rohrer_data <= t1_h)) * 100
pct_t2_raw = np.mean((rohrer_data >= t2_l) & (rohrer_data <= t2_h)) * 100
out_right_raw = np.sum(rohrer_data > t2_h)
out_left_raw = np.sum(rohrer_data < t2_l)

# --- FUNKCJA DO STATYSTYK Z SZEREGÓW ROZDZIELCZYCH ---
def calc_grouped_stats(bins_edges, data):
    frequencies, edges = np.histogram(data, bins=bins_edges)
    h_arr = np.diff(edges) # Długości klas
    midpoints = edges[:-1] + h_arr/2
    n_g = np.sum(frequencies)
    
    if n_g == 0:
        return [np.nan]*15
        
    # Średnia
    mean_g = np.sum(midpoints * frequencies) / n_g
    # Odchylenie standardowe
    var_g = np.sum(frequencies * (midpoints - mean_g)**2) / (n_g - 1)
    std_g = np.sqrt(var_g)
    
    # Dominanta
    d_idx = np.argmax(frequencies)
    L_d = edges[d_idx]
    f_d = frequencies[d_idx]
    f_d_minus_1 = frequencies[d_idx-1] if d_idx > 0 else 0
    f_d_plus_1 = frequencies[d_idx+1] if d_idx < len(frequencies)-1 else 0
    h_d = h_arr[d_idx]
    
    denom = (f_d - f_d_minus_1) + (f_d - f_d_plus_1)
    mode_g = L_d + ((f_d - f_d_minus_1) / denom) * h_d if denom != 0 else L_d + h_d/2
    
    # Mediana
    cum_freq = np.cumsum(frequencies)
    m_idx = np.where(cum_freq >= n_g/2)[0][0]
    L_m = edges[m_idx]
    F_m_minus_1 = cum_freq[m_idx-1] if m_idx > 0 else 0
    f_m = frequencies[m_idx]
    h_m = h_arr[m_idx]
    median_g = L_m + ((n_g/2 - F_m_minus_1) / f_m) * h_m if f_m != 0 else L_m + h_m/2
    
    # Błędy
    err_mean_g = std_g / np.sqrt(n_g)
    err_std_g = std_g / np.sqrt(2*n_g)
    err_median_g = 1.2533 * err_mean_g
    # Błąd dominanty dla szeregu (przybliżenie)
    err_mode_g = np.nan 
    
    # Zmienność i asymetria
    cv_g = (std_g / mean_g) * 100
    skew_g_mixed = 3 * (mean_g - median_g) / std_g if std_g != 0 else np.nan
    
    # Do asymetrii klasycznej z szeregu potrzebny jest 3 moment centralny, tu pomijamy na rzecz mieszanego lub uzywamy wzoru (Śr-D)/s
    skew_g_class = (mean_g - mode_g) / std_g if std_g != 0 else np.nan
    
    # Brak wartości odstających liczymy z próby nieuporządkowanej, dla szeregu wstawiamy 'x' jak w tabeli
    return [
        round(mean_g, 2), round(err_mean_g, 3), round(std_g, 2), round(err_std_g, 3), 
        round(median_g, 2), round(err_median_g, 3), round(mode_g, 2), err_mode_g, 
        round(cv_g, 2), round(skew_g_class, 2), round(skew_g_mixed, 2), 
        'x', 'x', 'x', 'x'
    ]

# --- PARAMETRY SZEREGÓW ---
# 1. "Najlepszy" (Reguła Sturgesa)
k_sturges = int(np.ceil(1 + 3.322 * np.log10(n)))
r = np.max(rohrer_data) - np.min(rohrer_data)
h_best = r / k_sturges
x01_best = np.min(rohrer_data) - 0.01 # minimalne przesunięcie
bins_best = [x01_best + i*h_best for i in range(k_sturges + 1)]
# Upewnienie się, że ostatni przedział zamyka dane
bins_best[-1] = max(bins_best[-1], np.max(rohrer_data) + 0.01)

# 2. Zmienione h (np. zwiększone o 50%)
h_changed = h_best * 1.5
k_changed = int(np.ceil(r / h_changed))
bins_h_changed = [x01_best + i*h_changed for i in range(k_changed + 1)]
bins_h_changed[-1] = max(bins_h_changed[-1], np.max(rohrer_data) + 0.01)

# 3. Zmienione x01 (przesunięte w lewo o pół klasy)
x01_changed = x01_best - (h_best * 0.5)
bins_x01_changed = [x01_changed + i*h_best for i in range(k_sturges + 2)]
bins_x01_changed[-1] = max(bins_x01_changed[-1], np.max(rohrer_data) + 0.01)

# 4. Klasyfikacja Lundmana (standardowe progi dla wskaźnika Rohrera)
# Smukła < 1.12, Średnia 1.13 - 1.25, Krępa > 1.25. (Przyjęte przybliżone granice).
bins_lundman = [np.min(rohrer_data)-0.1, 11.2, 12.5, np.max(rohrer_data)+0.1]
# UWAGA: Wartości wskaźnika Rohrera zależą od jednostek. Jeśli Masa[kg]/Wzrost[m]^3 to średnia jest ok. 12. 
# Więc Lundman dla kg/m^3 to: smukła < 11.2, średnia 11.3 - 12.5, krępa > 12.5
bins_lundman = [np.min(rohrer_data)-0.1, 11.2, 12.5, np.max(rohrer_data)+0.1]


# Zbieranie wyników
results = {
    'próba nieuporządkowana': [
        round(mean_raw, 2), round(err_mean_raw, 3), round(std_raw, 2), round(err_std_raw, 3), 
        round(median_raw, 2), round(err_median_raw, 3), round(mode_raw, 2), np.nan, 
        round(cv_raw, 2), round(skew_raw, 2), round(mixed_skew_raw, 2), 
        round(pct_t1_raw, 2), round(pct_t2_raw, 2), out_right_raw, out_left_raw
    ],
    'szereg "najlepszy"': calc_grouped_stats(bins_best, rohrer_data),
    'zmienione h': calc_grouped_stats(bins_h_changed, rohrer_data),
    'zmienione x_01': calc_grouped_stats(bins_x01_changed, rohrer_data),
    'klasyfikacja Lundmana': calc_grouped_stats(bins_lundman, rohrer_data)
}

index_names = [
    'średnia', 'błąd średniej', 'odch. Standard.', 'bład odchylenia', 
    'mediana', 'bład mediany', 'dominanta', 'bład dominanty', 
    'wsp. zmienności', 'wsp. asymetrii', 'mieszany wsp.asymetrii', 
    'procent typowe1', 'procent typowe2', 
    'liczba wartości odstających po prawej stronie', 'liczba wartości odstających po lewej stronie'
]

df_res = pd.DataFrame(results, index=index_names)
df_res.to_excel('wyniki_zad2.xlsx')
print("Sukces! Wygenerowano plik 'wyniki_zad2.xlsx'")