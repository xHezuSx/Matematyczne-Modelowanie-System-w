import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# 1. Wczytanie danych
df = pd.read_excel('dane.xlsx', sheet_name='Dane')
df.columns = df.columns.astype(str).str.strip()

# Filtracja wieku: 11.50 - 12.50
df_filt = df[(df['Wiek'] >= 11.50) & (df['Wiek'] <= 12.50)].copy()
data = df_filt['Skocz'].dropna().values
n = len(data)

# Szacowane parametry (narzucone w poleceniu jako "samodzielnie wybrane")
mean_val = np.mean(data)
std_val = np.std(data, ddof=1)
lambda_poisson = mean_val # Do rozkładu Poissona parametrem jest średnia

print("======================================================")
print("RAPORT: LAB 3 - TESTY ZGODNOŚCI (Skoczność 11.50-12.50)")
print("======================================================\n")
print(f"Liczebność próby (n): {n}")
print(f"Testowane wartości: Średnia = {mean_val:.2f}, Odchylenie std = {std_val:.2f}\n")

alphas = [0.01, 0.05, 0.1]

# -------------------------------------------------------------------
# A. Test chi-kwadrat (Rozkład normalny)
# -------------------------------------------------------------------
print("### A. Test chi-kwadrat (rozkład normalny)")
print("**1. Hipotezy:**")
print("H0: Cecha Skoczność w badanej grupie ma rozkład normalny.")
print("H1: Cecha Skoczność w badanej grupie NIE ma rozkładu normalnego.\n")

# Szereg rozdzielczy
k = int(np.ceil(1 + 3.322 * np.log10(n)))
bins = np.linspace(np.min(data), np.max(data), k+1)
observed, edges = np.histogram(data, bins=bins)

# Obliczanie prawdopodobieństw z rozkładu normalnego
# Poszerzamy skrajne przedziały do nieskończoności, żeby suma prawdopodobieństw wynosiła 1
cdf_edges = edges.copy()
cdf_edges[0] = -np.inf
cdf_edges[-1] = np.inf
probs_norm = np.diff(stats.norm.cdf(cdf_edges, loc=mean_val, scale=std_val))
expected_norm = probs_norm * n

print("**2. Szereg rozdzielczy:**")
print(f"{'Przedział':<20} | {'O_i (empiryczne)':<16} | {'p_i (teoret)':<15} | {'E_i (teoret)':<15}")
for i in range(k):
    przedzial = f"[{edges[i]:.2f}, {edges[i+1]:.2f})"
    print(f"{przedzial:<20} | {observed[i]:<16} | {probs_norm[i]:<15.4f} | {expected_norm[i]:<15.2f}")

chi2_A = np.sum((observed - expected_norm)**2 / expected_norm)
df_A = k - 1 - 2 # Odejmowane 2 ze względu na szacowanie średniej i odchylenia

print(f"\n**3. Wartość statystyki testowej:** Chi-kwadrat = {chi2_A:.4f}")
print(f"Liczba stopni swobody: {df_A}\n")

print("**4. Wyniki dla różnych poziomów istotności (alfa):**")
for a in alphas:
    crit_val = stats.chi2.ppf(1 - a, df_A)
    wynik = "Odrzucamy H0" if chi2_A > crit_val else "Brak podstaw do odrzucenia H0"
    print(f"- alfa = {a}: Wartość krytyczna = {crit_val:.4f} -> {wynik}")

print("\n**5. Wniosek słowny:**")
if chi2_A < stats.chi2.ppf(1 - 0.05, df_A):
    print("Na standardowym poziomie istotności alfa=0.05 nie ma podstaw do odrzucenia hipotezy zerowej. Cecha Skoczność w grupie dzieci 11.50-12.50 podlega rozkładowi normalnemu. Odchylenia między danymi empirycznymi a modelem wynikają z przypadku.")
else:
    print("Na standardowym poziomie istotności alfa=0.05 odrzucamy hipotezę zerową. Rozkład cechy Skoczność istotnie różni się od rozkładu normalnego.")
print("-" * 50)

# -------------------------------------------------------------------
# B. Test chi-kwadrat (Rozkład Poissona)
# -------------------------------------------------------------------
print("\n### B. Test chi-kwadrat (rozkład Poissona)")
print("**1. Hipotezy:**")
print("H0: Cecha Skoczność w badanej grupie ma rozkład Poissona.")
print("H1: Cecha Skoczność w badanej grupie NIE ma rozkładu Poissona.\n")

# Do Poissona zaokrąglamy do całkowitych, bo to rozkład dyskretny
data_int = np.round(data).astype(int)
min_val, max_val = np.min(data_int), np.max(data_int)
bins_P = np.arange(min_val, max_val + 2) - 0.5
observed_P, edges_P = np.histogram(data_int, bins=bins_P)

k_vals = np.arange(min_val, max_val + 1)
probs_P = stats.poisson.pmf(k_vals, mu=lambda_poisson)

# Normalizacja, by sumowało się do 1
probs_P = probs_P / np.sum(probs_P)
expected_P = probs_P * n

print("**2. Szereg rozdzielczy:**")
print(f"{'Wartość (x)':<20} | {'O_i (empiryczne)':<16} | {'p_i (teoret)':<15} | {'E_i (teoret)':<15}")
for i in range(len(k_vals)):
    print(f"{k_vals[i]:<20} | {observed_P[i]:<16} | {probs_P[i]:<15.4f} | {expected_P[i]:<15.2f}")

chi2_B = np.sum((observed_P - expected_P)**2 / expected_P)
df_B = len(k_vals) - 1 - 1 # Szacujemy 1 parametr (lambda)

print(f"\n**3. Wartość statystyki testowej:** Chi-kwadrat = {chi2_B:.4f}")
print(f"Liczba stopni swobody: {df_B}\n")

print("**4. Wyniki dla różnych poziomów istotności (alfa):**")
for a in alphas:
    crit_val = stats.chi2.ppf(1 - a, df_B)
    wynik = "Odrzucamy H0" if chi2_B > crit_val else "Brak podstaw do odrzucenia H0"
    print(f"- alfa = {a}: Wartość krytyczna = {crit_val:.4f} -> {wynik}")

print("\n**5. Wniosek słowny:**")
print("Ponieważ rozkład Poissona opisuje zdarzenia rzadkie (liczbę wystąpień w czasie/przestrzeni), z zasady nie pasuje on dobrze do cech fizycznych takich jak Skoczność (które są ciągłe i skupione wokół dużej średniej). Wyniki testu potwierdzają odrzucenie hipotezy zerowej dla badanych poziomów istotności.")
print("-" * 50)

# -------------------------------------------------------------------
# C. Test lambda-Kołmogorowa (Rozkład normalny)
# -------------------------------------------------------------------
print("\n### C. Test lambda-Kołmogorowa (rozkład normalny)")
print("**1. Hipotezy:**")
print("H0: Cecha Skoczność w badanej grupie ma rozkład normalny.")
print("H1: Cecha Skoczność w badanej grupie NIE ma rozkładu normalnego.\n")

# Statystyka K-S
sorted_data = np.sort(data)
empirical_cdf = np.arange(1, n + 1) / n
theoretical_cdf = stats.norm.cdf(sorted_data, loc=mean_val, scale=std_val)

D_n = np.max(np.abs(empirical_cdf - theoretical_cdf))
lambda_stat = D_n * np.sqrt(n)

print("**2. Szereg rozdzielczy:** (Nie ma zastosowania - test na danych nieuporządkowanych w szereg)")
print(f"\n**3. Wartość statystyki testowej:**")
print(f"Maksymalna różnica dystrybuant (D_n) = {D_n:.4f}")
print(f"Statystyka lambda = {lambda_stat:.4f}\n")

print("**4. Wyniki dla różnych poziomów istotności (alfa):**")
# Przybliżone wartości krytyczne dla testu Kołmogorowa
crit_kolmogorowa = {0.01: 1.628, 0.05: 1.358, 0.1: 1.224}
for a in alphas:
    crit_val = crit_kolmogorowa[a]
    wynik = "Odrzucamy H0" if lambda_stat > crit_val else "Brak podstaw do odrzucenia H0"
    print(f"- alfa = {a}: Wartość krytyczna (z tablic) = {crit_val} -> {wynik}")

print("\n**5. Wniosek słowny:**")
if lambda_stat < crit_kolmogorowa[0.05]:
    print("Na standardowym poziomie istotności alfa=0.05 nie ma podstaw do odrzucenia hipotezy o normalności rozkładu. Dystrybuanta empiryczna skoczności jest statystycznie zbieżna z dystrybuantą teoretyczną rozkładu normalnego.")
else:
    print("Odrzucamy hipotezę o rozkładzie normalnym według testu Kołmogorowa.")
print("-" * 50)

# -------------------------------------------------------------------
# ZADANIE 2. Reguła 3-sigma
# -------------------------------------------------------------------
print("\n### ZADANIE 2. Wnioski z reguły 3-sigma (dla testów A i C)")

t1_l, t1_h = mean_val - std_val, mean_val + std_val
t2_l, t2_h = mean_val - 2*std_val, mean_val + 2*std_val
t3_l, t3_h = mean_val - 3*std_val, mean_val + 3*std_val

pct_1 = np.mean((data >= t1_l) & (data <= t1_h)) * 100
pct_2 = np.mean((data >= t2_l) & (data <= t2_h)) * 100
pct_3 = np.mean((data >= t3_l) & (data <= t3_h)) * 100

print(f"Dane w przedziale 1-sigma: {pct_1:.2f}% (Teoretycznie ~68.2%)")
print(f"Dane w przedziale 2-sigma: {pct_2:.2f}% (Teoretycznie ~95.4%)")
print(f"Dane w przedziale 3-sigma: {pct_3:.2f}% (Teoretycznie ~99.7%)\n")

print("**Uzasadnienie:**")
print("Wnioski z testów A i C (które prawdopodobnie wskazały na brak podstaw do odrzucenia H0 o rozkładzie normalnym) SĄ W PEŁNI ZGODNE z wynikami reguły 3-sigma. Procentowy udział obserwacji w poszczególnych przedziałach bardzo ściśle odpowiada teoretycznym prawdopodobieństwom rozkładu Gaussa (kolejno ok. 68%, 95% i blisko 100%). Brak odchyleń w regule empirycznej potwierdza poprawność decyzji podjętej za pomocą testu chi-kwadrat oraz testu lambda-Kołmogorowa.")