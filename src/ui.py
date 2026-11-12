"""Интерфейс Better call LLM. Запуск: python -m streamlit run app.py."""

import atexit

import streamlit as st

from src.audio_capture import list_sources
from src.config import Settings, load_settings
from src.worker import SolverWorker


@st.cache_data(ttl=10, show_spinner=False)
def audio_devices():
    return list_sources()


def release_worker(worker: SolverWorker) -> None:
    atexit.unregister(worker.close)
    worker.close()


@st.cache_resource(scope="global", on_release=release_worker)
def get_worker() -> SolverWorker:
    settings = load_settings()
    worker = SolverWorker()
    worker.conversation.pause = settings.extractor_pause
    worker.conversation.maximum = settings.extractor_maximum
    worker.task_stability = settings.task_stability
    worker.audio_buffer_seconds = settings.audio_buffer_seconds
    worker.asr_queue_seconds = settings.asr_queue_seconds
    worker.speech_fragment_seconds = settings.speech_fragment_seconds
    worker.speech_silence_seconds = settings.speech_silence_seconds
    atexit.register(worker.close)
    return worker


def edit_condition(text: str) -> None:
    st.session_state.pending_task_text = text
    st.session_state.editor_open = True
    st.rerun(scope="app")


@st.fragment(run_every=0.5)
def capture_panel(worker: SolverWorker) -> None:
    state = worker.capture.snapshot()
    # Новая вкладка подключается к общему захвату, а не отзывает согласие
    # из-за исходного unchecked-состояния собственного виджета.
    if state.running:
        st.session_state.capture_consent = True
    mic = monitor = None
    with st.expander("Аудиоустройства и согласие", expanded=not st.session_state.get("capture_consent", False)):
        if st.button("Обновить устройства", key="refresh_devices", disabled=state.running):
            audio_devices.clear()
        try:
            sources, default_mic, default_monitor = audio_devices()
            microphones = [source.name for source in sources if not source.monitor]
            monitors = [source.name for source in sources if source.monitor]
            descriptions = {source.name: source.description for source in sources}
            if not microphones or not monitors:
                st.warning("Подключите микрофон и наушники, затем обновите устройства.")
            else:
                left, right = st.columns(2)
                with left:
                    mic = st.selectbox("Твой микрофон", microphones, key="capture_mic", disabled=state.running,
                                       index=microphones.index(default_mic) if default_mic in microphones else 0,
                                       format_func=descriptions.get)
                with right:
                    monitor = st.selectbox("Собеседник — выход на наушники", monitors, key="capture_monitor", disabled=state.running,
                                           index=monitors.index(default_monitor) if default_monitor in monitors else 0,
                                           format_func=descriptions.get)
        except Exception:
            st.warning("Не удалось получить устройства. Проверьте доступ к PipeWire/PulseAudio.")
        def consent_changed() -> None:
            if not st.session_state.capture_consent:
                worker.stop_capture()

        consent = st.checkbox("Все участники согласны на обработку разговора", key="capture_consent",
                              on_change=consent_changed)
        st.caption("Аудио распознаётся на ноутбуке. Текст отправляется в Codex; аудиозаписи не сохраняются.")
    controls, automatic = st.columns(2)
    with controls:
        if state.running:
            if st.button("Остановить прослушивание", key="stop_capture", width="stretch"):
                worker.stop_capture()
        elif st.button("Начать прослушивание", key="start_capture", type="primary", width="stretch",
                       disabled=not consent or mic is None or monitor is None):
            worker.start_capture(mic, monitor, consent=consent)
    with automatic:
        if st.button("Авторешение: включено" if worker.auto_enabled else "Авторешение: выключено",
                     key="toggle_auto", width="stretch", disabled=worker.context_broken):
            worker.set_auto(not worker.auto_enabled)
        st.caption("Полное условие запускает решение автоматически.")
    state = worker.capture.snapshot()
    live = worker.live_snapshot()
    status = "Распознаём речь…" if live.status == "recognizing" else "Слушаем разговор" if state.running else "Прослушивание остановлено"
    st.caption(status)
    left, right = st.columns(2)
    with left:
        st.progress(float(state.microphone_level), text="Ты")
    with right:
        st.progress(float(state.system_level), text="Собеседник")
    for error in (state.error, live.error):
        if error:
            st.error(error)
    if worker.context_broken:
        st.warning("Часть разговора пропущена. Начните новый контекст и повторите условие.")
        if st.button("Начать новый контекст", key="reset_audio_context"):
            worker.reset_context()


@st.fragment(run_every=0.5)
def condition_panel(worker: SolverWorker) -> None:
    state = worker.conversation.snapshot()
    st.subheader("Условие")
    labels = {"waiting": "Ожидаем задачу", "pending": "Накапливаем реплики…", "extracting": "Уточняем условие…",
              "ready": "Условие готово", "incomplete": "Нужно уточнение", "error": "Ошибка выделения условия"}
    st.caption("Решаем с оговорками" if state.status == "ready" and state.task.ambiguities else labels[state.status])
    if state.task.condition:
        st.markdown(state.task.condition)
        if st.button("Исправить условие", key="edit_extracted"):
            edit_condition(state.task.solver_condition)
    else:
        problem = worker.snapshot().problem
        if problem:
            st.markdown(problem)
        else:
            st.caption("Прочитайте задачу вслух — её условие появится здесь.")
    for ambiguity in state.task.ambiguities:
        st.warning(ambiguity)
    if state.error:
        st.error(state.error)


def manual_editor(worker: SolverWorker, settings: Settings, backend: str) -> None:
    with st.expander("Ручной ввод и повторное решение", expanded=st.session_state.get("editor_open", False)):
        st.text_area("Условие задачи", key="task_text", height=190,
                     placeholder="Введите задачу или нажмите «Исправить условие».", on_change=worker.cancel)
        st.caption("Ручная правка приостанавливает авторешение. Формулы можно писать в LaTeX.")
        label = "Решить заново" if worker.snapshot().problem else "Решить"
        if st.button(label, key="solve", type="primary", width="stretch",
                     disabled=backend == "OpenAI API" and not settings.openai_api_key):
            if st.session_state.task_text.strip():
                worker.set_auto(False)
                worker.start(st.session_state.task_text, settings, use_codex=backend == "Codex CLI")
            else:
                st.warning("Введите условие задачи.")


@st.fragment(run_every=0.5)
def solution_panel(worker: SolverWorker) -> None:
    state = worker.snapshot()
    st.subheader("Решение")
    labels = {"waiting": "Ожидание", "solving": "Решаем…", "done": "Готово", "error": "Ошибка"}
    st.caption(labels[state.status])
    if state.status == "solving" and st.button("Остановить решение", key="stop_solution"):
        worker.cancel()
    if state.text:
        if state.status == "waiting":
            st.warning("Это предыдущий или остановленный ответ.")
        st.markdown(state.text)
    elif state.status == "solving":
        st.info("Решение появится здесь. Прослушивание продолжается.")
    else:
        st.caption("После полного условия здесь появится решение.")
    if state.error:
        st.error(state.error)
    if state.problem:
        with st.expander("Для какого условия получен ответ"):
            st.markdown(state.problem)
    st.caption("Проверьте рассуждения и вычисления — модель может ошибаться.")


@st.fragment(run_every=0.5)
def transcript_panel(worker: SolverWorker) -> None:
    live = worker.live_snapshot()
    with st.container(height=280):
        if not live.replicas:
            st.caption("Распознанные реплики появятся после начала прослушивания.")
        for replica in live.replicas:
            st.caption(f"{replica.speaker} · {replica.start:.1f}–{replica.end:.1f} с")
            st.write(replica.text)


@st.fragment(run_every=0.5)
def audio_panel(worker: SolverWorker) -> None:
    audio = st.file_uploader("Короткая запись на русском", key="audio_file",
                             type=["wav", "mp3", "m4a", "ogg", "flac", "opus", "webm", "mp4"])
    state = worker.speech_snapshot()
    if st.button("Распознать", key="recognize", disabled=audio is None or state.status == "recognizing"):
        worker.start_transcription(audio.getvalue(), audio.name)
        state = worker.speech_snapshot()
    if state.status == "recognizing":
        st.info("Распознавание…")
    elif state.status == "done":
        st.caption(f"{state.filename} · {state.seconds:.2f} с")
        if state.text:
            st.write(state.text)
            if st.button("Использовать как условие", key="use_transcript"):
                edit_condition(state.text)
        else:
            st.info("Речь не обнаружена.")
    elif state.error:
        st.error(state.error)
    st.caption("Распознавание файла локальное, без вызова решателя.")


@st.fragment(run_every=0.5)
def diagnostics_panel(worker: SolverWorker) -> None:
    capture, live = worker.capture.snapshot(), worker.live_snapshot()
    st.write(f"Блоки: микрофон {capture.microphone_blocks}, системный звук {capture.system_blocks}")
    st.write(f"Очередь: {live.raw_blocks} блоков, {live.fragments} фрагментов · Задержка ASR {live.backlog_seconds:.1f} с")
    st.write(f"Потери: {live.dropped_blocks} блоков, {live.dropped_fragments} фрагментов")
    state = worker.conversation.snapshot()
    st.write(f"Проверок экстрактора: {state.checks}")
    st.text("\n".join(state.replicas) or "Контекст пока пуст.")


def main() -> None:
    st.set_page_config(page_title="Better call LLM", page_icon="∑", layout="wide")
    st.markdown("""<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1440px;}
    h1 {letter-spacing: -0.025em;}
    </style>""", unsafe_allow_html=True)
    st.title("Better call LLM")
    st.caption("Слушает разговор. Собирает условие. Помогает решить.")
    try:
        settings = load_settings()
    except ValueError as error:
        st.error(str(error))
        return
    with st.expander("Настройки решателя"):
        backend = st.selectbox("Решатель для ручного запуска", ["Codex CLI", "OpenAI API"], key="solver_backend")
        st.caption("Авторешение и выделение условия используют Codex CLI и его лимиты.")
        if backend == "OpenAI API":
            if not settings.openai_api_key:
                st.warning("Добавьте OPENAI_API_KEY в .env.")
            st.caption(f"Модель: {settings.openai_model}. Запросы расходуют API-кредиты.")
    worker = get_worker()
    st.session_state.setdefault("task_text", "")
    if "pending_task_text" in st.session_state:
        st.session_state.task_text = st.session_state.pop("pending_task_text")
        worker.cancel()
    with st.container(border=True):
        capture_panel(worker)
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        with st.container(border=True):
            condition_panel(worker)
        manual_editor(worker, settings, backend)
    with right:
        with st.container(border=True):
            solution_panel(worker)
    with st.expander("Расшифровка разговора"):
        transcript_panel(worker)
    with st.expander("Дополнительно"):
        files, replicas, diagnostics = st.tabs(["Аудиофайл", "Текстовые реплики", "Диагностика"])
        with files:
            audio_panel(worker)
        with replicas:
            with st.form("replica_form", clear_on_submit=True):
                speaker = st.selectbox("Кто говорит", ["Ты", "Собеседник"], key="replica_speaker")
                text = st.text_input("Новая реплика", key="replica_text", max_chars=2000)
                if st.form_submit_button("Добавить реплику") and text.strip():
                    worker.add_replica(text, speaker)
        with diagnostics:
            diagnostics_panel(worker)


if __name__ == "__main__":
    main()
