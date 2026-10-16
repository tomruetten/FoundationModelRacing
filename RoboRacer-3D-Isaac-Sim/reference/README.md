================================================================
reference — Kalibrierung & Streckendaten
================================================================

calibration/
─────────────
camera_intrinsics.yaml               Kamera-Intrinsics (OAK-D Pro W,
                                      fx/fy/cx/cy + Verzeichnungskoeffizienten)
extrinsics_candidate_checked_v2.yaml Kamera→Fahrzeug-Extrinsics (geprüfte
                                      Kandidatenlösung)
README_sensor_tf_usage.md            Wie die obigen TFs in eigenem Code
                                      verwendet werden (Frames, Konventionen)

raceline_FTM_Halle/
────────────────────
track_FTM_Halle.json                 Streckengeometrie (Mittellinie,
                                      Breite) der FTM-Halle-Strecke
reftraj_FTM_Halle_min_time.json      Minimalzeit-Referenztrajektorie auf
                                      dieser Strecke (Quelle: Ziad Elnaggar) —
                                      wird von simulation/closed_loop/17_raceline_
                                      video_config_D.py und
                                      16_raceline_retest_correct_checkpoint.py
                                      geladen (siehe deren Docstrings für
                                      das JSON-Format).
