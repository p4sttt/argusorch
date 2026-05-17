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
- [x] **Алгоритмы TD(n) и GAE (Generalized Advantage Estimation):** 
  - Реализация расчета **TD(0)** ошибки для критика.
  - Экспоненциальное сглаживание n-шаговых возвратов `TD(λ)`.
  - Реализация вычисления преимущества $\hat{A}_t$ с параметрами $\gamma$ (discount factor) и $\lambda$ (GAE parameter).
- [x] **Алгоритм оптимизации стратегии (PPO / MAPPO):** 
  - Клиппирование отношения вероятностей (Probability Ratio Clipping) в `MAACLoss`.

## 🔄 В процессе (In Progress)
- [ ] **Архитектура LLM Actor-Critic:** 
  - Инициализация акторов (генерация действий/токенов).
  - Инициализация централизованного критика, принимающего на вход `joint_state`.
- [ ] **Обновление весов и интеграция (Updater):**
  - Интеграция с PEFT/LoRA для эффективного дообучения LLM-весов (Actor/Critic optimizers).
  - Реализация батчевого прохода в `CentralizedCritic.evaluate_states()`.
- [ ] **Сбор траекторий (Rollout Buffer):** Накопление данных (состояния, действия, награды, логарифмы вероятностей) для оффлайн/онлайн апдейтов.

## 📝 Осталось реализовать (To Do)
- [ ] **Мониторинг метрик:** Логирование TD-Error, Critic Loss, Policy Loss и Policy Entropy для отладки.