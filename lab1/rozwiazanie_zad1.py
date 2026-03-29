import pandas as pd
import numpy as np
import io

# 1. Wczytanie danych
data = """8.04        9.14        7.46        6.58
6.95        8.14        6.77        5.76
7.58        8.74        12.74       7.71
8.81        8.77        7.11        8.84
8.33        9.26        7.81        8.47
9.96        8.1         8.84        7.04
7.24        6.13        6.08        5.25
4.26        3.1         5.39        12.5
10.84       9.13        8.15        5.56
4.82        7.26        6.42        7.91
5.68        4.74        5.73        6.89"""

df = pd.read_csv(io.StringIO(data), sep=r'\s+', header=None, names=['Zestaw 1', 'Zestaw 2', 'Zestaw 3', 'Zestaw 4'])

results = {}

# 2. Obliczenia dla każdego zestawu
for col in df.columns:
    x = df[col]
    n = len(x)
    mean_x = x.mean()
    
    var_pop = x.var(ddof=0)
    std_pop = x.std(ddof=0)
    
    var_samp = x.var(ddof=1)
    std_samp = x.std(ddof=1)
    
    # Klasyczne wartości typowe
    t1_low, t1_high = mean_x - std_samp, mean_x + std_samp
    t2_low, t2_high = mean_x - 2*std_samp, mean_x + 2*std_samp
    
    pct_t1 = ((x >= t1_low) & (x <= t1_high)).sum() / n * 100
    pct_t2 = ((x >= t2_low) & (x <= t2_high)).sum() / n * 100
    
    # Wartości odstające (klasyczne)
    outliers_t2 = x[(x < t2_low) | (x > t2_high)].tolist()
    outliers_str = ", ".join(map(str, outliers_t2)) if outliers_t2 else "Brak"
    
    cv = (std_samp / mean_x) * 100
    
    # Parametry pozycyjne (Kwantyle)
    median_x = x.median()
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    
    q_dev = (q3 - q1) / 2
    
    # Kwartylowe wartości typowe
    qt1_low, qt1_high = median_x - q_dev, median_x + q_dev
    qt2_low, qt2_high = q1 - 3*q_dev, q3 + 3*q_dev
    
    # Kwartylowe wartości odstające
    q_outliers = x[(x < qt2_low) | (x > qt2_high)].tolist()
    q_outliers_str = ", ".join(map(str, q_outliers)) if q_outliers else "Brak"
    
    cv_q = (q_dev / median_x) * 100
    
    # Asymetria
    if q3 != q1:
        skew_q = ((q3 - median_x) - (median_x - q1)) / (2 * (q3 - q1))
    else:
        skew_q = 0
        
    skew_m = 3 * (mean_x - median_x) / std_samp
    
    # 3. Zapis do słownika wynikowego
    results[col] = {
        'Liczebność próby': n,
        'Średnia': round(mean_x, 2),
        'Wariancja populacji': round(var_pop, 2),
        'Odchylenie standardowe populacji': round(std_pop, 2),
        'Wariancja z próby': round(var_samp, 2),
        'Odchylenie standardowe z próby': round(std_samp, 2),
        'Typowe 1 (przedział)': f"[{t1_low:.2f}, {t1_high:.2f}]",
        'Procent wartości typowe 1 (%)': round(pct_t1, 2),
        'Typowe 2 (przedział)': f"[{t2_low:.2f}, {t2_high:.2f}]",
        'Procent wartości typowe 2 (%)': round(pct_t2, 2),
        'Wartości odstające': outliers_str,
        'Klasyczny współczynnik zmienności (%)': round(cv, 2),
        'Mediana': round(median_x, 2),
        'Kwartyl pierwszy': round(q1, 2),
        'Kwartyl trzeci': round(q3, 2),
        'Odchylenie ćwiartkowe': round(q_dev, 2),
        'Kwartylowe typowe 1 (przedział)': f"[{qt1_low:.2f}, {qt1_high:.2f}]",
        'Kwartylowe typowe 2 (przedział)': f"[{qt2_low:.2f}, {qt2_high:.2f}]",
        'Kwartylowe wartości odstające': q_outliers_str,
        'Kwartylowy współczynnik zmienności (%)': round(cv_q, 2),
        'Kwartylowy współczynnik asymetrii': round(skew_q, 2),
        'Mieszany współczynnik asymetrii': round(skew_m, 2)
    }

# 4. Eksport do Excela
res_df = pd.DataFrame(results)
res_df.to_excel('wyniki_lab1.xlsx')
print("Pomyślnie wygenerowano plik 'wyniki_lab1.xlsx' z pełnymi statystykami!")