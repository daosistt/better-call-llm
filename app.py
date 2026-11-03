"""Минимальный интерфейс Math Copilot. Запуск: python -m streamlit run app.py."""

import streamlit as st

from config import load_settings
from worker import SolverWorker


@st.cache_resource(scope="session", on_release=lambda worker: worker.close())
def get_worker() -> SolverWorker:
    return SolverWorker()


@st.fragment(run_every=0.5)
def audio_panel(worker: SolverWorker) -> None:
    st.subheader("Распознавание аудиофайла")
    audio = st.file_uploader(
        "Короткая запись на русском",
        type=["wav", "mp3", "m4a", "ogg", "flac", "opus", "webm", "mp4"],
        key="audio_file",
    )
    state = worker.speech_snapshot()
    if st.button("Распознать", key="recognize", disabled=audio is None or state.status == "recognizing"):
        worker.start_transcription(audio.getvalue(), audio.name)
        state = worker.speech_snapshot()
    if state.status == "recognizing":
        st.info("Распознавание…")
    elif state.status == "done":
        st.caption(f"Файл: {state.filename} · Обработка: {state.seconds:.2f} с")
        if state.text:
            st.text(state.text)
            if st.button("Использовать как условие", key="use_transcript"):
                st.session_state.pending_task_text = state.text
                st.rerun(scope="app")
        else:
            st.info("Речь не обнаружена.")
    elif state.error:
        st.error(state.error)
    st.caption("Аудио обрабатывается локально и хранится только в памяти. Время включает загрузку модели при первом запуске.")


@st.fragment(run_every=0.5)
def solution_panel(worker: SolverWorker) -> None:
    state = worker.snapshot()
    labels = {"waiting": "Ожидание", "solving": "Решение…", "done": "Завершено", "error": "Ошибка"}
    st.write(f"Статус: {labels[state.status]}")
    if state.text:
        if state.status == "waiting":
            st.caption("Предыдущий или остановленный ответ. Запустите решение актуального условия.")
        st.markdown(state.text)
    if state.error:
        st.error(state.error)
    st.caption("Ответ модели может содержать ошибки — проверьте рассуждения и вычисления.")


def main() -> None:
    st.set_page_config(page_title="Math Copilot", page_icon="∑")
    st.title("Math Copilot")
    st.caption("Локальный помощник для математических задач")

    settings = load_settings()
    if settings.openai_api_key:
        st.caption(f"API-ключ задан · Модель: {settings.openai_model}")
    else:
        st.warning("Добавьте OPENAI_API_KEY в .env в папке проекта.")

    if "task_text" not in st.session_state:
        st.session_state.task_text = ""

    worker = get_worker()
    if "pending_task_text" in st.session_state:
        st.session_state.task_text = st.session_state.pop("pending_task_text")
        worker.cancel()
    st.text_area(
        "Условие задачи",
        key="task_text",
        height=200,
        placeholder="Введите задачу. Формулы можно записывать в LaTeX.",
        on_change=worker.cancel,
    )
    st.caption("При запуске условие отправится в OpenAI API. Используются API-кредиты.")
    state = worker.snapshot()
    label = "Решить заново" if state.problem else "Решить"
    if st.button(label, key="solve", type="primary", disabled=not settings.openai_api_key):
        if st.session_state.task_text.strip():
            worker.start(st.session_state.task_text, settings)
        else:
            st.warning("Введите условие задачи.")
    if st.button("Остановить решение"):
        worker.cancel()
    solution_panel(worker)
    audio_panel(worker)


if __name__ == "__main__":
    main()
