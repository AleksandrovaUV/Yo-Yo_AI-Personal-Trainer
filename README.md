# Yo-Yo: AI Personal_Trainer
In this project, we create a functional software that allows to have an in-house yoga practise, using adanced Machine Learning technologies.

## About

**Name:**  
**Topic:**  
**Author**: Aleksandrova Uliana  
**Scientific Supervisor:** Vinogradov Andrei  
**Organization:** RUDN University: Peoples' Friendship University of Russia. Department of Artifitial Intelligence and Machine Learning.  
**Type:** Final qualifying work  
**DOI:**  
**Year:**  
**Language**: Russian, English  
**More About This Project:** [Wiki Page](https://github.com/AleksandrovaUV/Yo-Yo_AI-Personal-Trainer/wiki)

# Content
Проект посвящён разработке и исследованию модели классификации поз йоги на основе графовых нейронных сетей (GCN). В качестве входных данных используются 23 ключевые точки человеческого тела, полученные из полуавтоматически размеченного набора изображений. Модель обучается на нормализованных позах, аннотированных вручную по методу полу-автоматической разметки данных, и классифицирует изображения по заранее определённым категориям (например: Tree, Warrior, Bound Angle, Chair и др.).

Проект включает:

* компактный набор ключевых точек (23 точки), оптимизированный для задач классификации поз;  
* модуль нормализации позы (центрирование, масштабирование, ориентация);  
* функции аугментации позы (повороты, шум, масштабирование);  
* графовую нейронную сеть (GCN) для классификации;  
* пайплайн обучения и валидации;  
* экспериментальное сравнение качества модели.

## Структура репозитория 

```
YO-YO_AI-PERSONAL-TRAINER/
│
├── .venv/                          # виртуальное окружение (локальное)
│
├── annotated_images/               # изображения с разметкой (результаты)
│
├── data/                           # подготовленные данные (CSV, NPZ и др.)
│
├── data_annotation/                # модуль полуавтоматической разметки
│   ├── annotation_pipeline.py
│   ├── experiment_benchmark.py
│   ├── image_marker.py
│   ├── interactive_module.py       # интерактивный аннотатор (23 точки)
│   ├── iterative_module.py         # модуль правил и проверки позы
│   ├── procession_module.py
│   ├── analysis.ipynb
│   ├── experiment_models.json
│   ├── experiment.json
│   ├── manual_annotation.json      # итоговая ручная разметка
│   ├── preannotations.json         # автоматическая предразметка
│   └── saved_model.pb
│
├── model_0.0/                      # старая версия модели
│
├── model_0.1/
│   ├── visual_train/
│   │   ├── data_procession.ipynb
│   │   ├── data_to_csv.py
│   │   ├── gcn_dataset_test.npz
│   │   ├── gcn_dataset_train.npz
│   │   └── gcn_dataset_valid.npz
│   ├── ... (другие файлы модели)
│
├── movenet/                        # модель MoveNet (весы, графы)
│
├── prepared_data/                  # изображения, подготовленные к разметке
│
├── .gitignore
├── graph_opt.pb
├── hpe_test_ez
├── LICENSE
├── README.md                       # будет заменён на новый
├── test_data.csv
├── TEST_DELETE.py
├── train_data.csv
└── valid_data.csv
```

## Структура проекта

### Разметка данных (`data_annotation/`)
- `interactive_module.py` — интерактивный аннотатор (23 точки, zoom, pan, drag).  
- `iterative_module.py` — модуль правил (validity, anatomy, angles, proportions, topology).  
- `preannotations.json` — автоматическая предразметка (MediaPipe).  
- `manual_annotation.json` — итоговая разметка после исправлений.  
- `experiment_benchmark.py` — оценка качества предразметки.  
- `annotation_pipeline.py` — полный пайплайн разметки.

### Модель (`model_0.1/`)
- `visual_train/` — подготовка данных, NPZ‑файлы датасета.  
- `gcn_dataset_train.npz`, `gcn_dataset_valid.npz`, `gcn_dataset_test.npz` — данные для GCN.  
- `data_to_csv.py` — преобразование данных.  
- Jupyter‑ноутбуки для анализа.

### Данные
- `prepared_data/` — изображения для разметки.  
- `annotated_images/` — изображения с нанесёнными точками.  
- `train_data.csv`, `valid_data.csv`, `test_data.csv` — CSV‑версии датасета.


## Запуск проекта

### 1. Установка зависимостей
```
pip install -r requirements.txt
```

### 2. Запуск аннотатора
```
python data_annotation/interactive_module.py
```

Режимы:
- `f` — свободная разметка (с нуля),
- `e` — исправление изображений с ошибками (error‑based).

В режиме `e` доступна кнопка **Check**, запускающая повторную проверку аномалий.

### 3. Подготовка данных для модели
Файлы `.npz` уже лежат в `model_0.1/visual_train/`.


### 5. Оценка модели
```
python model_0.1/data_procession.ipynb
```
## Результаты первичного тестирования

### Реализовано
- полуавтоматическая разметка (предразметка + интерактивная коррекция),
- модуль поиска аномалий (6 групп правил),
- кнопка **Check** для повторной проверки,
- нормализация позы (центр, масштаб, ориентация),
- аугментации (шум, поворот, масштаб),
- GCN‑классификатор поз,
- метрики PCP, PCK, OKS,
- визуализация и анализ.

### Запланировано
- temporal‑GCN для видео,
- автоматический подбор гиперпараметров,
- экспорт модели в ONNX,
- улучшенная визуализация ошибок.


