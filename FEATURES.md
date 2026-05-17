# ArgusOrch: MARL CTDE Library Features

Библиотека для мультиагентного обучения с подкреплением (MARL) на базе парадигмы CTDE (Centralized Training with Decentralized Execution) для языковых моделей (LLM).
Основано на идеях CoMLRL, но с расширенной теоретической базой для оценки функции ценности.

## ✅ Готово (Implemented)

- [x] **Базовая инфраструктура сред:** Класс `MultiAgentTextEnv` для пошагового взаимодействия текстовых агентов.
- [x] **Поддержка CTDE:**
  - Децентрализованное исполнение (Decentralized Execution) через генерацию локальных наблюдений `_generate_observations()`.
  - Централизованное обучение (Centralized Training) благодаря методу `joint_state()`, собирающему глобальный контекст для Критика.
- [x] **Тестовые среды:**
  - `CodeCollabEnv` (Плотная награда / Dense Rewards: Coder + Tester).
  - `LongHorizonPlanningEnv` (Редкая награда / Sparse Rewards: планирование с наградой в конце эпизода).
- [x] **Архитектура LLM Actor-Critic:**
  - Инициализация акторов (генерация действий/токенов).
  - Инициализация централизованного критика, принимающего на вход `joint_state`.
- [x] **Алгоритмы TD(n) и GAE (Generalized Advantage Estimation):**
  - Реализация расчета **TD(0)** ошибки для критика.
  - Экспоненциальное сглаживание n-шаговых возвратов `TD(λ)`.
  - Реализация вычисления преимущества $\hat{A}_t$ с параметрами $\gamma$ (discount factor) и $\lambda$ (GAE parameter).
- [x] **Алгоритм оптимизации стратегии (PPO / MAPPO):**
  - Клиппирование отношения вероятностей (Probability Ratio Clipping) в `MAACLoss`.
- [x] **Обновление весов и интеграция (Updater):**
  - Реализация батчевого прохода в `CentralizedCritic.evaluate_states()` с Autograd.
  - Добавлено клиппирование градиентов (Gradient Clipping).
  - Интеграция с PEFT/LoRA для эффективного дообучения LLM-весов (применено в examples).
- [x] **Сбор траекторий (Rollout Buffer):** Накопление данных в `ReplayBuffer` и формирование батчей `TrainingBatch` для On-Policy RL апдейтов.

## 🔄 В процессе (In Progress)

## 📝 Осталось реализовать (To Do)

- [ ] **Мониторинг метрик:** Логирование TD-Error, Critic Loss, Policy Loss и Policy Entropy для отладки.
