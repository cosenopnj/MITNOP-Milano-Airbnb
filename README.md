# Airbnb Milano — Analiza i detekcija netipičnih oglasa

**Tim:** Teodora Šović (IN 7/2023), Luka Kostić (IN 32/2023), Rastko Penić (IN 54/2023)  
**Šifra projekta:** Airbnb Milano

---

## Opis projekta

Projekat analizira tržište kratkoročnog iznajmljivanja smeštaja u Milanu putem platforme Airbnb. Koriste se dva skupa podataka — jun 2025. i septembar 2025. Projekat obuhvata četiri faze:

1. **Faza 1 i 2** — Čišćenje podataka, istraživačka analiza i vizualizacija (Teodora Šović)
2. **Faza 3** — Predviđanje godišnjeg prihoda oglasa primenom MLP neuralne mreže (Rastko Penić)
3. **Faza 4** — Detekcija anomalnih oglasa primenom autoenkodera (Luka Kostić)

---

## Struktura projekta

```
MITNOP-Milano-Airbnb/
├── README.md
└── code/
    ├── 01_data_preparation_and_analysis.py   # Faza 1 i 2 — čišćenje i analiza
    ├── 03_mlp.py                              # Faza 3 — MLP za predviđanje prihoda
    └── 04_autoencoder_anomaly.py              # Faza 4 — autoenkoder za detekciju anomalija
```

---

## Podaci

Podaci se preuzimaju sa platforme **InsideAirbnb**: https://insideairbnb.com/get-the-data/

Potrebna su dva fajla za grad **Milano, Lombardy, Italy**:

| Fajl | Datum snimka | Sačuvati kao |
|------|-------------|--------------|
| `listings.csv.gz` | Jun 2025 (19.06.2025.) | `listings_jun.csv` |
| `listings.csv.gz` | Septembar 2025 (22.09.2025.) | `listings_september.csv` |

Oba fajla moraju biti smeštena u isti folder kao Python skripte pre pokretanja.

---

## Instalacija zavisnosti

Projekat zahteva **Python 3.12**. Sve potrebne biblioteke instaliraju se jednom komandom:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras folium scipy
```

---

## Redosled pokretanja

### Korak 1 — Čišćenje i analiza podataka
```
01_data_preparation_and_analysis.py
```
**Ulaz:** `listings_jun.csv`, `listings_september.csv`  
**Izlaz:**
- `listings_jun_clean.csv` — očišćen jun skup
- `listings_sept_clean.csv` — očišćen septembar skup
- `mapa_jun.html` — interaktivna toplomerna karta gustine oglasa za jun
- `mapa_sept.html` — interaktivna toplomerna karta gustine oglasa za septembar
- Grafovi: histogram cena, box-plot po kvartovima, korelaciona matrica, t-test, udeo tipova smeštaja

> ⚠️ Ovaj korak mora biti izvršen prvi jer generišeodčišćene CSV fajlove koji su potrebni za Korake 2 i 3.

---

### Korak 2 — Predviđanje prihoda (MLP)
```
03_mlp.py
```
**Ulaz:** `listings_jun_clean.csv`, `listings_sept_clean.csv`  
**Izlaz:**
- `mlp_model.keras` — sačuvani model
- Grafovi: scatter plot predviđenih vs. stvarnih prihoda (jun i septembar), train vs. val loss
- Ispis RMSE metrika na konzoli (ukupni, ispod 20.000 € i ispod 50.000 €)

---

### Korak 3 — Detekcija anomalija (Autoenkoder)
```
04_autoencoder_anomaly.py
```
**Ulaz:** `listings_jun_clean.csv`, `listings_sept_clean.csv`  
**Izlaz:**
- `anomaly_map_jun.html` — interaktivna karta anomalnih oglasa za jun
- `anomaly_map_sept.html` — interaktivna karta anomalnih oglasa za septembar
- Grafovi: kriva učenja autoenkodera, histogram greške rekonstrukcije sa pragom
- Ispis top 10 anomalnih oglasa na konzoli

---

## Napomene

- Skripte su razvijane i testirane u **Spyder 6** razvojnom okruženju
- Svi fajlovi (Python skripte i CSV podaci) moraju se nalaziti u **istom folderu**
- Interaktivne mape (`.html` fajlovi) otvaraju se u web pregledaču
- Autoenkoder koristi nasumičnu inicijalizaciju težina — rezultati mogu neznatno varirati između pokretanja
- Faza 3 čuva istrenirani model kao `mlp_model.keras` koji se može učitati za kasniju upotrebu
