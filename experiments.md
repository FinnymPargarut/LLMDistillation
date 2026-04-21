# Подробное описание экспериментов

## Experiment_00 Sanity-check

Проверка на вменяемость модели gemma-3-1b-it.
Батчевый инференс на dev subset (DEV_SIZE=200) датасета HuggingFaceTB/Countdown-Task-GOLD

Аргументы вызова:
sys.argv = [
    'script.py',
    '--n', '100',
    '--out', 'outputs/exp_00',
    '--max-new-tokens', '512',
    '--greedy',
    '--verbose-first', '5'
]

Общая сводка эксперимента:
 {'dataset_len': 200, 'equation_extracted': 21, 'equation_correct': 1, 'extracted_rate': 0.105, 'accuracy': 0.005, 'avg_response_chars': 650.01}

Результат:
Модель смогла написать ответ только на 10% примеров, в остальных случаях она ловила ограничение на максимальное число токенов.
Гемма часто путала числа (добавляла новые и убирала изначальные), писала неправильные ответы на свои вычисления.

## Experiment_01 Check teacher on dev

Говоря прямо, это та же проверка на вменяемость, что и в прошлом эксперименте, но для учителя (Qwen3-8B).
Проверяем, чтобы все работало с vLLM на A100 YC для следующего этапа (для experiment_02).
Хотел сначала генерировать ответы на kaggle 2xT4, но тогда ухудшается качество ответов из-за 4bit квантования, 
а скорость ужасно медленная - передумал.

Вызов:
!PYTHONPATH=. python3 scripts/exp_01_teacher_check.py --model-id Qwen/Qwen3-8B --tp 1 --dtype bfloat16 --exp-name exp01_teacher_check_200

Общая сводка эксперимента:
{
"n_examples": 200,
"equation_extracted": 171,
"equation_correct": 162,
"extracted_rate": 0.855,
"accuracy": 0.81,
"avg_response_tokens": 2487.45
}

Результаты:
Скорость инференса и точность ответов хороши. Было сложно, но с адом зависимостей Yandex Datasphere разобрался.
Далее если упрусь в стенку, можно будет проанализировать, где модели ошибаются (результаты каждой итерации сохранены в json).

## Experiment_02 Inference teacher on train

Генерируем ответы учителя (Qwen3-8B) на train subset. Как и в предыдущем эксперименте используем vLLM с YC A100.

Вызов:
!PYTHONPATH=. python3 scripts/exp_02_generate_teacher_data.py \
    --model-id Qwen/Qwen3-8B \
    --tp 1 \
    --dtype bfloat16 \
    --start-idx 0 \
    --n-examples 1000 \
    --output-name teacher_chunk_02
Генерировал три раза: 0-1000 (по времени 19 минут), 1000-3000(заняло 35 минут), 3000-10000 (116 минут).

Общая сводка по чанкам:
{
  "n_examples": 1000,
  "equation_extracted": 876,
  "equation_correct": 844,
  "extracted_rate": 0.876,
  "accuracy": 0.844,
  "avg_response_tokens": 2122.13,
  "start_idx": 0,
  "end_idx": 1000,
  "elapsed_s": 1010.3236048221588,
  "model_id": "Qwen/Qwen3-8B"
},
{
  "n_examples": 2000,
  "equation_extracted": 1787,
  "equation_correct": 1698,
  "extracted_rate": 0.8935,
  "accuracy": 0.849,
  "avg_response_tokens": 2083.395,
  "start_idx": 1000,
  "end_idx": 3000,
  "elapsed_s": 1963.0084428787231,
  "model_id": "Qwen/Qwen3-8B"
},
Генерация 3000-10000 идет прямо сейчас.

Результаты:
Очень хорошая точность. Как можно заметить, почти все извлекаемые ответы правильные (accuracy очень близка к extracted_rate).
Также по выводам модели можно заметить, что около половины правильных ответов генерируются при низком количествв ответных токенов
(< 1000, часто 200-500). avg_response_tokens такой высокий, потому что его тянут вверх долгие генерации квена, утыкающиеся в лимит.
Даже несмотря на то, что лимит был увеличен до 8192 токенов, модель все еще часто утыкается в него.