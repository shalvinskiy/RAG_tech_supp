# agent_lokis

Корпоративный QA-пайплайн: сравнение **zero-shot LLM** и **RAG (LangChain + FAISS)** на закрытых внутренних документах.

Датасет: **10 статей**, **50 вопросов** с эталонными ответами (`data/`).

---

## 1. Подходы

### 1.1 Zero-shot (без RAG)

Модель получает только вопрос + системный промпт («ответь кратко / если не знаешь — скажи»).  
Корпоративных регламентов в контексте нет → типичный ответ: отказ или галлюцинация.

### 1.2 RAG (Retrieval-Augmented Generation)

Обязательные компоненты (LangChain):


| Компонент     | Реализация                                                    |
| ------------- | ------------------------------------------------------------- |
| **Индекс**    | чанкинг статей → эмбеддинги → FAISS (`src/rag/index.py`)      |
| **Ретривер**  | top-k семантический поиск по индексу                          |
| **Генератор** | Gemini API или локальная LLM через Ollama (OpenAI-compatible) |


Схема:

```
вопрос → FAISS retriever (top_k) → контекст + вопрос → LLM → ответ
```

Промпт RAG: отвечать **только по контексту**; если факта нет — сказать, что информации недостаточно.

---



## 2. Модели и стек


| Роль                 | Модель / инструмент                                           | Где крутится             |
| -------------------- | ------------------------------------------------------------- | ------------------------ |
| Генератор (API)      | `gemini-3.5-flash-lite`                                       | Google Gemini API        |
| Генератор (local)    | `qwen2.5:1.5b` (по умолчанию)                                 | Ollama, endpoint `/v1`   |
| Judge (LLM-as-judge) | Gemini (`JUDGE_MODEL`)                                        | API                      |
| Embeddings           | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | CPU, локально            |
| Vector store         | FAISS                                                         | `indexes/faiss_articles` |


---



## 3. Метрики


| Метрика                         | Что измеряет                                                       |
| ------------------------------- | ------------------------------------------------------------------ |
| **BLEU** (sacrebleu)            | n-gram                                                             |
| **ROUGE-1 / ROUGE-2 / ROUGE-L** | unigram / bigram / LCS overlap                                     |
| **LLM-as-judge**                | Gemini оценивает факты 0..5 → в summary также `llm_judge_mean_0_5` |


Лексические метрики чувствительны к формулировке (короткий «160 евро.» vs полный эталон). Для фактологии полезнее **LLM-as-judge** (запуск без `--no-judge`).

### 3.1 Результаты пилота (n=3, gemini-3.5-flash-lite, без judge)

Одинаковые первые вопросы датасета (`q1`–`q3`), `top_k=3`.


| Режим         | BLEU      | ROUGE-1   | ROUGE-2 | ROUGE-L   |
| ------------- | --------- | --------- | ------- | --------- |
| **Zero-shot** | 0.046     | 0.000     | 0.000   | 0.000     |
| **RAG**       | **0.567** | **0.889** | 0.000   | **0.889** |


Файлы:

- `results/zeroshot_gemini_20260728_184356.json`
- `results/rag_gemini_20260728_184006.json`



### 3.2 Выводы пилота

1. **Zero-shot почти бесполезен** на закрытых корпоративных фактах: модель честно отказывает («нет в знаниях») → лексические метрики ≈ 0.
2. **RAG резко поднимает качество**: на q1/q2 ответы фактически верные (TravelHub — 5 рабочих дней; лимит РФ — 9000 ₽).
3. **ROUGE-2 = 0 при высоком ROUGE-1** — артефакт коротких русских формулировок / токенизации, не обязательно «плохой ответ».
4. Ретривер в целом попадает в нужную статью (командировки), но иногда подмешивает соседние чанки (SLA, удалёнка) — есть запас по улучшению retrieval.
5. Полный прогон на **50 вопросах** + **LLM-as-judge** + сравнение **Ollama** ещё нужно запустить командами из §6 (в пилоте только Gemini, n=3).

Качественный пример (RAG, q1):

- GT: *«Заявку нужно подать не позднее чем за 5 рабочих дней до выезда.»*
- Pred: *«Заявку на командировку в TravelHub нужно подать не позднее чем за 5 рабочих дней до выезда.»*

---



## 4. Структура репозитория

```
agent_lokis/
├── data/
│   ├── articles.json       # 10 корпоративных статей
│   ├── questions.json      # 50 вопросов
│   └── ground_truth.json   # эталонные ответы
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── models/             # gemini / ollama клиенты
│   ├── metrics/            # BLEU/ROUGE + LLM-judge
│   └── rag/                # индекс + pipeline
├── scripts/
│   ├── build_index.py
│   ├── evaluate_zeroshot.py
│   ├── evaluate_rag.py
│   ├── test.py
│   └── compare_results.py
├── notebooks/
│   ├── ask_rag.ipynb           # один вопрос в RAG
│   └── ollama_setup.ipynb      # проверка Ollama
├── indexes/                    # FAISS (генерируется)
├── results/                    # JSON-отчёты
├── .env.example
└── requirements.txt
```

---



## 5. Установка и конфигурация

Интерпретатор: `/usr/bin/python3` (системный). Ставить только недостающие пакеты.

```bash
cd /home/romanrussia/Code_git/agent_lokis
cp .env.example .env
# прописать GEMINI_API_KEY

# при необходимости:
/usr/bin/python3 -m pip install --user -r requirements.txt
```

Ключевые переменные `.env`:


| Переменная        | Описание                                        |
| ----------------- | ----------------------------------------------- |
| `GEMINI_API_KEY`  | ключ Google AI                                  |
| `GEMINI_MODEL`    | генератор, по умолчанию `gemini-3.5-flash-lite` |
| `JUDGE_MODEL`     | модель судьи                                    |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434/v1`                     |
| `OLLAMA_MODEL`    | id модели Ollama, напр. `qwen2.5:1.5b`          |
| `TOP_K`           | число чанков в контексте                        |
| `EMBEDDING_MODEL` | модель эмбеддингов                              |


**Не коммитьте** `.env` **с ключом.**

---



## 6. Как запускать

Все команды из корня репозитория.

### 6.1 Smoke-test (zero-shot, один вопрос)

```bash
/usr/bin/python3 scripts/test.py --backend gemini --question-id q1
/usr/bin/python3 scripts/test.py --backend ollama --question-id q1
```



### 6.2 Сборка индекса

```bash
/usr/bin/python3 scripts/build_index.py
# опции: --chunk-size 600 --chunk-overlap 100
```



### 6.3 Zero-shot evaluation

```bash
# быстрый пилот
/usr/bin/python3 scripts/evaluate_zeroshot.py --backend gemini --limit 3 --no-judge

# полный прогон + LLM-as-judge
/usr/bin/python3 scripts/evaluate_zeroshot.py --backend gemini
/usr/bin/python3 scripts/evaluate_zeroshot.py --backend ollama
```



### 6.4 RAG evaluation

```bash
/usr/bin/python3 scripts/build_index.py   # если индекса ещё нет

/usr/bin/python3 scripts/evaluate_rag.py --backend gemini --limit 3 --no-judge
/usr/bin/python3 scripts/evaluate_rag.py --backend gemini --top-k 3
/usr/bin/python3 scripts/evaluate_rag.py --backend ollama
```



### 6.5 Сравнение отчётов

```bash
/usr/bin/python3 scripts/compare_results.py results/zeroshot_*.json results/rag_*.json
```



### 6.6 Один вопрос в RAG (ноутбук)

Открыть `notebooks/ask_rag.ipynb` и выполнить ячейки, либо:

```python
from src.models.factory import get_llm
from src.rag.pipeline import RAGPipeline

llm = get_llm("gemini")
rag = RAGPipeline(llm=llm, top_k=3)
result = rag.answer("Ваш вопрос...")
print(result.answer)
print([h.title for h in result.hits])
```



### 6.7 Локальная LLM через Ollama

1. Установить [Ollama](https://ollama.com/download).
2. Запустить и скачать модель:

```bash
ollama serve
ollama pull qwen2.5:1.5b
```

1. Проверить в `notebooks/ollama_setup.ipynb` или сразу прописать в `.env`:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
OLLAMA_API_KEY=ollama
OLLAMA_MODEL=qwen2.5:1.5b
```

1. Локально:

```bash
/usr/bin/python3 scripts/evaluate_zeroshot.py --backend ollama --limit 5
/usr/bin/python3 scripts/evaluate_rag.py --backend ollama --limit 5
```

Для честного сравнения zero-shot vs RAG используйте **один и тот же** `--backend`.

---



## 7. Как улучшить пайплайн

1. **Лучшие эмбеддинги** — `intfloat/multilingual-e5-large` / `BAAI/bge-m3` вместо MiniLM.
2. **Гибридный поиск** — BM25 + dense (например, EnsembleRetriever) для точных чисел и названий.
3. **Чанкинг** — подбор `chunk_size` / overlap; parent-document / small-to-big retrieval.
4. **Reranker** — cross-encoder поверх top-20 → top-3.
5. **Дедуп чанков** — в пилоте иногда дважды попадала одна статья.
6. **Промпт генератора** — жёстче требовать полный ответ в формате эталона (цифры + единицы).
7. **Метрики** — полный n=50 + LLM-as-judge; отдельно **retrieval metrics** (Hit@k, MRR) по `article_id`

