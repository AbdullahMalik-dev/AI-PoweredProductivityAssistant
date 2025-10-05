# AI-Powered Productivity Assistant
An AI-powered productivity assistant using reinforcement learning to optimize study breaks.

⚠️ I'm a student and still learning. This project is a work-in-progress.

This is a student-built productivity assistant that uses **Reinforcement Learning (RL)** to optimize study and break schedules. It's designed to help improve focus, reduce distractions, and build better habits — one decision at a time.

---

## What It Does

- Monitors a simulated “focus level”
- Learns when to **Keep Studying**, **Take a Break**, or **Adjust Study Strategy**
- Uses the PPO algorithm (from Stable-Baselines3) to train a model that balances productivity and rest
- Plots rewards, actions, and focus levels for visualization

---
## Features

- **Reinforcement Learning Agent**: Uses PPO to decide optimal study/break actions based on recent focus history.
- **Custom Focus Environment**: Simulates attention drops and rewards smart break-taking.
- **Screen Usage Monitor**: Logs top apps and compares changes in usage behavior.
- **AFK Notifier**: Sends a notification if your activity drops too low.
- **Text Summarizer + Keyword Extractor**: Add your notes and get key concepts + a summary using `BERT` and `YAKE`.
- **Modular Design**: Cleanly split into `ActiveModel.py`, `notificationModel.py`, and the main logic.

  ## File Overview
    `main.py` — Core logic for training, monitoring, summarizing, and RL agent actions.

   `ActiveModel.py` — Custom-built module to detect whether the user is active.

    `notificationModel.py` — Simplified wrapper around a notification system.

## Example of an Output:
```
-----------------------
Text to analyze for key concepts: [waits 5 seconds]

Study Guide Summary:
  [Generated summary from BART]

Key Concepts in this Text:
- Concept 1
- Concept 2
...

AI Decision: Take a Break
Top Apps Used: [('chrome.exe', 18), ('discord.exe', 12), ('code.exe', 9)]
```
