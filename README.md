***********************
# PROTOTYPICAL NETWORKS
***********************

# ZZSN gr. 12:
- Kamil Dąbrowski
- Łukasz Pietraszek


Założenie projektowe
--------------------
Sieci prototypowe opierają się na założeniu, że istnieje osadzanie, w którym kilka punktów skupia się wokół pojedynczej reprezentacji prototypu dla każdej klasy. Jego celem jest nauka prototypów dla poszczególnych klas na podstawie uśredniania próbek w przestrzeni cech.

Zbiory danych
-------------
### Omniglot
Zestaw danych Omniglot jest przeznaczony do opracowywania podobnych do ludzkiego uczenia się algorytmów uczenia się. Zawiera 1623 różnych odręcznych znaków z 50 różnych alfabetów. Każda z 1623 symboli został narysowany online przez 20 różnych ludzi.

### Mini-imagenet 
Zestaw danych mini-ImageNet został zaproponowany przez Vinyals et al. do oceny uczenia się za pomocą kilku strzałów. Jego złożoność jest wysoka ze względu na użycie obrazów ImageNet, ale wymaga mniej zasobów i infrastruktury niż praca na pełnym zestawie danych ImageNet. W sumie istnieje 100 klas z 600 próbkami kolorowych obrazów 84×84 na klasę. Te 100 klas podzielono odpowiednio na 64, 16 i 20 klas dla zadań próbkowania do meta-treningu, meta-walidacji i meta-testu. 

Instalacja
----------
Po sklonowaniu repozytorium tworzenie środowisko:
```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ pip install -e .
```

Uruchamianie eksperymentów
--------------------------
Eksperymenty przeprowadzono dla datasetów Omniglot i Mini-imagenet zgodnie z opisanymi w artykule parametrami. Modele porównano pod względem skuteczności treningu dla 1-shot (5-way Acc.), 5-shot (5-way Acc.), 1-shot (20-way Acc.), 5-shot (20-way Acc.). Do treningu modelu dla zbioru danych Omniglot wykorzystano 600 epok, każdy epizod trenujący zawierał 60 klas i po 5 query points na każdą klasę. Natomiast dla zbioru Mini-imagenet wybrano 64 klasy trenujące, 16 klas walidacyjne i 20 klas testujących.

Uruchomienie następuje poprzez:
```
$ python train.py -d dataset[omniglot/mini_imagenet]
```

Parametry eksperymentów
-----------------------
Parametry można ustawić w config/env.py:
#### GENERAL PARAMS:
* ```LERNING_RATE``` – szybkość uczenia, domyślnie 0.001
* ```GAMMA``` – gamma harmonogramu nauki modelu, domyślnie 0.5
* ```DECAY_EVERY``` – krok harmonogramu tempa nauki, domyślnie 20
* ```MAX_EPOCH``` – maksymalna liczba epok do wytrenowania domyślnie 500

#### NETWORK PARAMS:
* ```X_DIM``` – wymiar X sieci neuronowej, domyślnie (3, 84, 84)
* ```HID_DIM``` – wymiar HID sieci neuronowej, domyślnie 64
* ```Z_DIM``` – wymiar Z sieci neuronowej, domyślnie 64
* ```N_TEST``` – liczba uruchomień zbioru testowego, domyślnie 25

#### ALGORITHM PARAMS:
* ```NUM_WAY``` – droga, wykorzystane w eksperymentach 5 lub 20, domyślnie 20
* ```NUM_SHOT``` – strzały, wykorzystane w eksperymentach 1 lub 5, domyślnie 1
* ```NUM_QUERY``` – liczba próbek do wykorzystania na klasę w postaci zapytań walidacyjnych wykorzystane w eksperymentach 5 lub 15, domyślnie 5
* ```EPOCH_SIZE``` – liczba epok, domyślnie 100
* ```EPOCH_SIZE_TEST``` – liczba epok testującej, domyślnie 1000


Wyniki eksperymentów
--------------------
### Omniglot
<table>
  <tr>
    <th rowspan="2"></th>
    <th colspan="2">5-way</th>
    <th colspan="2">20-way</th>
  </tr>
  <tr>
    <th>1-shot</th>
    <th>5-shot</th>
    <th>1-shot</th>
    <th>5-shot</th>
  </tr>
  <tr>
    <td>Osiągnięte wyniki</td>
    <td>96,48 ± 0,08%</td>
    <td>98,92 ± 0,03%</td>
    <td>93,92 ± 0,05%</td>
    <td>98,29 ± 0,02%</td>
  </tr>
  <tr>
    <td>Wyniki w artykule</td>
    <td>98,8%</td>
    <td>99,7%</td>
    <td>96,0%</td>
    <td>98,9%</td>
  </tr>
</table>

![omniglot_hist](https://github.com/uuukaaasz/Prototypical_networks/blob/main/docs/omniglot_hist.png)

### Mini-imagenet
<table>
  <tr>
    <th rowspan="2"></th>
    <th colspan="2">5-way</th>
  </tr>
  <tr>
    <th>1-shot</th>
    <th>5-shot</th>
  </tr>
  <tr>
    <td>Osiągnięte wyniki</td>
    <td>39,63 ± 0,17%</td>
    <td>60,15 ± 0,11%</td>
  </tr>
  <tr>
    <td>Wyniki w artykule</td>
    <td>49,42 ± 0,78%</td>
    <td>68,20 ± 0,66%</td>
  </tr>
</table>

![mini_imagenet_hist](https://github.com/uuukaaasz/Prototypical_networks/blob/main/docs/mini_imagenet_hist.png)