================================================================
dynamics/retraining_20260815 — Nachtrainings-Experimente (2026-08-15)
================================================================

Fünf gezielte Nachtrainings-Varianten auf Basis der bestehenden GRU/MS-NN
Checkpoints, alle am selben Tag (2026-08-15) durchgeführt. Trainingslogs
sind nicht enthalten (Rohtext, wenig Mehrwert ohne die Original-Trainings-
daten) — bei Bedarf auf der Lab-Maschine unter
Zusammenfassung/dynamics/retraining_20260815/*.log verfügbar.

SKRIPTE (scripts/)
───────────────────
run_msnn_a1_widemargin_20260815.py       MS-NN, größere Klassifikations-
                                          marge ("wide margin")
run_msnn_a2_phase3_lowerlr_20260815.py   MS-NN, Phase 3 mit reduzierter
                                          Lernrate
run_training_run38_scheduled_sampling.py GRU Run38, scheduled sampling
                                          (graduelle Umstellung von
                                          Teacher-Forcing auf eigene
                                          Vorhersagen während Training)
run_training_run39_h200.py               GRU Run39, größere Hidden-Size
                                          (200 statt Standard)
run_training_run40_2layer.py             GRU Run40, 2-Layer-Architektur
                                          statt 1-Layer

RESULTIERENDE CHECKPOINTS (checkpoints/)
──────────────────────────────────────────
msnn_a1_widemargin_20260815/net.json
msnn_a1_widemargin_phase2_backup_20260815/net.json
msnn_a2_phase3_lowerlr_20260815/net.json
run38_schedsamp___20260815183802715201.pth
run39_h200___20260815185858062499.pth
run40_2layer___20260815191430922657.pth

STATUS
──────
Diese Varianten sind Explorationsergebnisse, keine der hier trainierten
Checkpoints hat den bestehenden Run30 (GRU) bzw. exp9 Phase2 (MS-NN)
als deployten Standard abgelöst — siehe dynamics/deployed/ für die
aktuell aktiven Checkpoints. Für Details zu Ergebnissen der einzelnen
Varianten: siehe Original-Logs auf der Lab-Maschine.
