import pandas as pd
import scipy.stats as stats

# Wczytanie i filtracja
df = pd.read_excel('dane.xlsx', sheet_name='Dane')
df.columns = df.columns.astype(str).str.strip()
df_12 = df[(df['Wiek'] >= 12.00) & (df['Wiek'] < 13.00)].copy()

# Obliczenie Rohrera
df_12['Wzrost_m'] = df_12['Wys'] / 100
df_12['Rohrer'] = df_12['Masa'] / (df_12['Wzrost_m']**3)

# Zebranie danych wspólnie i usunięcie wierszy z brakami w tych dwóch kolumnach
df_corr = df_12[['Rohrer', 'Skocz']].dropna()

# Korelacja Pearsona
correlation, p_value = stats.pearsonr(df_corr['Rohrer'], df_corr['Skocz'])

print(f"Współczynnik korelacji r-Pearsona: {correlation:.2f}")
if correlation < 0:
    print("Wynik ujemny potwierdza: im większy Rohrer, tym MNIEJSZA skoczność.")
else:
    print("Wynik dodatni: im większy Rohrer, tym WIĘKSZA skoczność.")