import asyncio
import sys
import traceback
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from langchain_gigachat import GigaChat
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv(Path(__file__).parent / ".env")

SERVER_SCRIPT = str(Path(__file__).parent.parent / "mcp_server" / "server.py")

SYSTEM_PROMPT = (
    "Ты — внимательный ИИ-ассистент по тестированию данных. Твоя задача — генерировать данные, строго следуя контексту диалога.\n"
    "КОНТЕКСТ ПРОЕКТА:\n"
    "- Все исходные файлы лежат в `data/raw/`.\n\n"
    "ПРАВИЛА ПОВЕДЕНИЯ:\n"
    "1. ВНИМАТЕЛЬНО ЧИТАЙ ИСТОРИЮ ЧАТА. Если пользователь уже назвал файл, НИКОГДА не переспрашивай его.\n"
    "2. Если пользователь просит указать точность (similarity), передай это в параметр `target_similarity`.\n"
    "3. Для чистых данных вызывай `generate_dataset`. Для искаженных: если данные нужно сгенерировать и исказить — `generate_custom_distortions`; если нужно исказить уже существующий файл (реальную выборку или ранее сгенерированный CSV) без генерации — `distort_existing_csv`.\n"
    "   Возможности искажений:\n"
    "   - маргинальные (по колонкам): typo, numeric_typo, outlier, missing, category_swap, noise_additive, noise_multiplicative, precision_loss;\n"
    "   - парные (между двумя колонками): break_correlation (разрушить/ослабить корреляцию до target_correlation в [-1,1]), false_link (создать ложную связь силой target_correlation в [-1,1], по умолчанию 0.5), invert_correlation (инвертировать знак), shift_dependency (сдвиг наклона линейной зависимости до target_slope); для них опционально method: 'rearrangement' (по умолчанию - распределение col2 сохраняется в точности) или 'mixture' (точная корреляция Пирсона, но форма распределения col2 меняется);\n"
    "   - структурные (таблица целиком): duplicate_rows, drop_column, rename_column, change_type, distribution_drift (дрейф распределения Y'=(Y+delta)*scale), violate_constraints (нарушение ограничений: constraint='unique' - создаёт дубликаты значений, 'range' - значения вне заявленного диапазона, требует min_value и/или max_value и ровно одну колонку, 'not_null' - NaN в обязательной колонке; rate - доля строк).\n"
    "   По умолчанию ячейки выбираются случайно (mechanism='ecar'). Если пользователь задает условие отбора строк (например, 'только у клиентов старше 40'), передай mechanism='enar' и condition — pandas-выражение по колонкам, например 'age > 40'.\n"
    "   Правила передавай в rules_json полным форматом: {\"marginal\": {колонка: {\"error_type\", \"error_rate\", \"mechanism\", \"condition\", \"intensity\"}}, \"pairwise\": [{\"col1\", \"col2\", \"action\", \"target_correlation\"?, \"target_slope\"?}], \"structural\": [{\"action\", \"columns\"?, \"rate\"?, \"constraint\"?, \"min_value\"?, \"max_value\"?}]}. Для силы шума используй intensity (доля std колонки), для точности — precision_loss с intensity=число знаков.\n"
    "   ГРУППОВАЯ АДРЕСАЦИЯ в 'marginal': одно правило можно применить к группе колонок — селектором '*' (все колонки) или '*numeric'/'*categorical'/'*boolean'/'*text'/'*datetime'/'*identifier' (все колонки данного типа), либо списком правил [{'columns': ['age', 'balance'], 'error_type': ...}]. Правила действуют в порядке следования, при пересечении последнее переопределяет предыдущее (удобно задавать исключения: сначала '*numeric', затем правило для конкретной колонки). Селектор, не соответствующий ни одной колонке, вернёт ошибку — тогда проверь типы колонок через profile_data.\n"
    "4. КОРРЕКТИРОВКА ПАРАМЕТРОВ: если в ответе generate_dataset achieved_similarity меньше target_similarity, изучи weak_columns и metrics_by_column: сообщи пользователю, какие колонки и по какой метрике отклоняются сильнее всего, затем скорректируй параметры и повтори generate_dataset (например, с max_attempts=10) без лишних вопросов.\n"
    "ПРАВИЛА ОФОРМЛЕНИЯ (СТРОГО):\n"
    "1. Отвечай живо (например, 'Я успешно сгенерировал...').\n"
    "2. Если в ответе инструмента есть поле 'target_similarity' (и оно не равно 0.9 по умолчанию), ОБЯЗАТЕЛЬНО напиши в начале: 'Был сгенерирован набор данных с точностью X% (достигнуто Y%)', подставив значения из полей 'target_similarity' и 'achieved_similarity'.\n"
    "3. Сделай ОДНУ комбинированную Markdown-таблицу из поля 'column_stats'. "
    "В таблице должны быть ТОЛЬКО столбцы: column, kind, n, uniq, is_unique, mean, std, min, max, top, top_share. "
    "Значения столбца 'column' имеют префиксы 'real_' (исходные данные) и 'synth_' (сгенерированные) - сохраняй их, чтобы было видно сравнение. "
    "Значения 'None' заменяй на прочерк '-'. "
    "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать теги <br> внутри ячеек таблицы.\n"
    "4. Сделай ВТОРУЮ Markdown-таблицу из поля 'sample_data' с первыми 10 строками.\n"
    "5. ЗАПРЕЩЕНО выводить сырой JSON. Используй только Markdown-таблицы.\n"
    "6. НЕ ИСПОЛЬЗУЙ смайлики."
)

async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT],
        env=None,
    )

    print("Подключение к MCP-серверу и GigaChat...")
    
    answers_dir = Path("data/chat_answers")
    answers_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                
                model = GigaChat(
                    model="GigaChat-2-Max",
                    streaming=False,
                    verify_ssl_certs=False,
                    timeout=120.0
                )
                
                agent = create_agent(model, tools, system_prompt=SYSTEM_PROMPT)
                
                print("Готово! Введите 'exit' для выхода.\n")

                messages_history = []
                msg_counter = 1

                while True:
                    try:
                        user_input = input("Вы: ")
                        if user_input.lower() in ("exit", "quit", "выход"):
                            break
                        if not user_input.strip():
                            continue

                        messages_history.append(HumanMessage(content=user_input))

                        print("GigaChat думает...")
                        response = await agent.ainvoke({"messages": messages_history})
                        
                        final_msg = response["messages"][-1]
                        final_content = getattr(final_msg, "content", "")
                        
                        messages_history.append(AIMessage(content=final_content))

                        timestamp = datetime.now().strftime("%H%M%S")
                        md_filename = answers_dir / f"answer_{msg_counter:03d}_{timestamp}.md"
                        
                        md_text = f"**Пользователь:** {user_input}\n\n---\n\n**GigaChat:**\n\n{final_content}"
                        md_filename.write_text(md_text, encoding="utf-8")
                        
                        msg_counter += 1

                        print(f"\nGigaChat (сохранено в {md_filename}):")
                        if final_content and str(final_content).strip():
                            print(final_content)
                        else:
                            print("[Пустой ответ от GigaChat]")
                        print("-" * 50)
                        
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        if "ReadTimeout" in str(e) or "Timeout" in str(e).__class__.__name__:
                            print("\n[Ошибка]: Превышено время ожидания ответа от сервера.")
                        else:
                            print(f"\nОшибка выполнения: {e}")
                            traceback.print_exc()
                        
                        messages_history.pop()

    except Exception as e:
        print(f"Критическая ошибка при запуске: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())