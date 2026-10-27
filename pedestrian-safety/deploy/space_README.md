---
title: Crowd Safety Console
emoji: 🚶
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Agentic AI early warning for crowd safety
---

# Crowd Safety Console

BITE308L Artificial Intelligence project, VIT. Shresth Pathak (24BIT0270), Siddharth Tiwari (23BIT0196). Guide: Dr. Srinivas Kopu.

Street sensors forecast each hour ahead, camera photos show how packed a place is now, and an AI agent (Gemini) combines both into one call: **Normal**, **Watch** or **Act now**.

- **Sensor model:** XGBoost on Melbourne pedestrian counts (next-hour crowd level)
- **Image model:** soft-voting ensemble of ViT, Xception and InceptionV3 hybrids on WiderPerson crops
- **Detector:** YOLO11s
- **Agent:** Gemini function calling over 6 tools, with a fixed-plan fallback

The first visit after a quiet spell can take a minute or two while the Space wakes up.
