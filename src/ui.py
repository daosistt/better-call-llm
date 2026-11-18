"""Интерфейс Better call LLM. Запуск: python -m streamlit run app.py."""

import atexit

import streamlit as st

from src.audio_capture import list_sources
from src.config import Settings, load_settings
from src.worker import SolverWorker
from src.prompt_loader import SOLVER_PROMPTS


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


@st.fragment(run_every=0.5)
def solver_prompt_panel(worker: SolverWorker) -> None:
    current = worker.solver_prompt
    if st.session_state.get("shown_solver_prompt") != current:
        st.session_state.solver_prompt = current
    st.session_state.shown_solver_prompt = current

    def choose_prompt() -> None:
        try:
            worker.set_solver_prompt(st.session_state.solver_prompt)
            st.session_state.pop("solver_prompt_error", None)
        except (OSError, ValueError):
            st.session_state.solver_prompt = worker.solver_prompt
            st.session_state.solver_prompt_error = "Не удалось прочитать промпт. Проверьте выбранный файл в prompts/solver/."

    st.selectbox("Промпт решателя", list(SOLVER_PROMPTS), format_func=SOLVER_PROMPTS.get,
                 key="solver_prompt", on_change=choose_prompt)
    if st.session_state.get("solver_prompt_error"):
        st.error(st.session_state.solver_prompt_error)
    st.caption("Общий выбор для ноутбука и телефона. Применяется к следующим решениям по кнопке и автоматическим решениям. "
               "Для нового ответа на прежнее условие нажмите «Решить заново».")


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
    status = ("Обработка приостановлена: нужно восстановить контекст" if worker.context_broken
              else "Распознаём речь…" if live.status == "recognizing"
              else "Слушаем разговор" if state.running else "Прослушивание остановлено")
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
        st.warning(f"{live.gap_reason or 'Часть разговора пропущена.'} Начните новый контекст и повторите условие.")
        st.caption("Кнопка сбросит неполный контекст и восстановит обработку речи. "
                   "Авторешение возобновится, если было включено до сбоя. "
                   "Если сам захват остановился, затем нажмите «Начать прослушивание».")
        if st.button("Начать новый контекст", key="reset_audio_context"):
            worker.reset_context()


@st.fragment(run_every=0.5)
def condition_panel(worker: SolverWorker, settings: Settings, backend: str) -> None:
    state = worker.conversation.snapshot()
    st.subheader("Условие")
    labels = {"waiting": "Ожидаем задачу", "pending": "Накапливаем реплики…", "extracting": "Уточняем условие…",
              "ready": "Условие готово", "incomplete": "Нужно уточнение", "error": "Ошибка выделения условия"}
    st.caption("Решаем с оговорками" if state.status == "ready" and state.task.ambiguities else labels[state.status])
    if state.task.condition:
        st.markdown(state.task.condition)
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

    solution = worker.snapshot()
    condition = state.task.solver_condition or solution.problem
    if st.button("Решить заново" if solution.problem else "Решить сейчас", key="solve",
                 type="primary", width="stretch",
                 disabled=not condition.strip() or worker.context_broken or solution.status == "solving"
                 or (backend == "OpenAI API" and not settings.openai_api_key)):
        worker.set_auto(False)
        worker.start(condition, settings, use_codex=backend == "Codex CLI")


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
def screen_panel(worker: SolverWorker) -> None:
    state = worker.screen_snapshot()
    st.caption("Снимается весь рабочий стол ноутбука, включая все мониторы. Снимок виден во всех вкладках этого сервера.")
    busy = state.status in {"capturing", "analyzing"}
    if st.button("Сделать снимок", key="capture_screen", width="stretch", disabled=busy):
        worker.start_screenshot()
        state = worker.screen_snapshot()
    if state.status == "capturing":
        st.info("Делаем снимок экрана ноутбука…")
    if state.image:
        st.image(state.image, caption=f"Экран ноутбука · {state.captured_at}", width="stretch")
        if st.button("Разобрать условие", key="analyze_screen", width="stretch",
                     disabled=state.status == "analyzing" or worker.context_broken):
            worker.start_screen_analysis()
            state = worker.screen_snapshot()
        if state.status == "analyzing":
            st.info("Выделяем условие со снимка и учитываем разговор…")
        elif state.analyzed:
            st.success("Снимок разобран. Результат — в карточке «Условие».")
    if state.error:
        st.error(state.error)
    if state.status in {"done", "error"}:
        if st.button("Убрать снимок", key="clear_screen"):
            worker.clear_screenshot()
            st.rerun(scope="app")
    st.caption("«Разобрать условие» отправляет показанный снимок и контекст разговора в Codex. "
               "Результат появится в карточке «Условие»; устные уточнения дополнят его.")


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
        backend = st.selectbox("Решатель для запуска по кнопке", ["Codex CLI", "OpenAI API"], key="solver_backend")
        st.caption("Авторешение и выделение условия используют Codex CLI и его лимиты.")
        if backend == "OpenAI API":
            if not settings.openai_api_key:
                st.warning("Добавьте OPENAI_API_KEY в .env.")
            st.caption(f"Модель: {settings.openai_model}. Запросы расходуют API-кредиты.")
    worker = get_worker()
    solver_prompt_panel(worker)
    with st.container(border=True):
        capture_panel(worker)
    with st.expander("Экран ноутбука"):
        screen_panel(worker)
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        with st.container(border=True):
            condition_panel(worker, settings, backend)
    with right:
        with st.container(border=True):
            solution_panel(worker)
    with st.expander("Расшифровка разговора"):
        transcript_panel(worker)


if __name__ == "__main__":
    main()
