src/ — Quelldateien (direkt aus ma_jonathanmohr-main/)
=======================================================
Hier liegen die Originalskripte und modifizierten Experimente.
Für Reproduktion: IMMER die nummerierten Skripte im übergeordneten
Ordner (11_train_msnn_exp9_best.py etc.) verwenden — die sind bereinigt.

MS-NN Experimente (unsere):
  exp9_withax_combined.py    ← ★ BESTES Modell (yr=0.1158)
  exp6c_combined_data.py     ← Paper-Protokoll P1+2+3 (yr=0.1196)
  exp8_combined_lowlr.py     ← noax Vergleich (yr=0.1191)
  exp10_three_sessions.py    ← 3-Session Vergleich (yr=0.1232)
  exp11_mirror_combined.py   ← Mirror-Versuch (FEHLGESCHLAGEN, yr=0.3060)

MS-NN Infrastruktur:
  inference_msnn.py          ← Originale Inference (erweiterte Version: 13_inference_msnn.py)
  train_msnn_full_car02.py   ← Basis MS-NN Trainingsskript von Felix

GRU:
  run_training_run30.py      ← Run30: 200ep/patience=30 (beste Konfiguration)
  RNN_architectures.py       ← GRU-Modellklassen (RNN_jomoh_1 etc.)
  training_v2.py             ← GRU-Trainingsloop (mit Dataset-Klassen)
  training.py                ← GRU-Trainingsloop (Original)

Daten:
  DataPreparation.py         ← CSV laden, filtern, normalisieren
  Dataset_Classes.py         ← PyTorch Dataset-Klassen für GRU
  process_all_sessions.py    ← Rohdaten → processed_data/
  generate_mirror_data.py    ← Mirror-Augmentierung (NUR für GRU!)

Evaluierung:
  post_training_eval.py      ← GRU cross-day eval nach Training
  eval_msnn_car02.py         ← MS-NN eval (alte Version)
  compile_results.py         ← Ergebnisse zusammenführen

Abhängigkeiten:
  requirements.txt           ← vollständige Abhängigkeiten
  requirements_reduced.txt   ← minimale Abhängigkeiten
