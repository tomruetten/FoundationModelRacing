================================================================
dynamics/deployed — Schlanke Inferenz-Skripte + deployte Checkpoints
================================================================

Im Unterschied zu dynamics/msnn/ (volle Trainings-/Experiment-Codebase)
enthält dieser Ordner nur das, was man braucht, um die beiden aktuell
deployten Modelle direkt zu benutzen — kein Training, nur Inferenz.

INHALT
──────
13_inference_msnn.py   MS-NN Inferenz (nnodely, CPU-only)
15_gru_inference.py    GRU Inferenz (torch)

trained_models/
  gru_run30_best.pth              GRU, 3-Kanal, deployter Checkpoint
                                   (Cross-Day-RMSE Day1: 0.4448 rad/s)
  msnn_exp9_phase2_backup/net.json  MS-NN exp9 Phase2, deployter Checkpoint
                                   (yr RMSE gleichtagig: 0.1158 rad/s)

VORAUSSETZUNGEN
────────────────
GRU:   conda env "roboracer" (torch, kein Extra-Setup nötig)
MS-NN: conda env "nnodely-env" (nnodely==1.5.4, CPU-only)
       conda create -n nnodely-env python=3.10 -y
       conda run -n nnodely-env pip install nnodely==1.5.4 numpy pandas torch scipy matplotlib

BENUTZUNG
─────────
Beide Skripte laden den jeweiligen Checkpoint aus trained_models/ und
erwarten eine Eingabe-CSV mit denselben Spalten wie die Original-
Trainingsdaten (siehe dynamics/training/README.md bzw.
dynamics/msnn/README.md für das Spaltenformat). Für eigene Daten:
Pfad zur CSV im jeweiligen Skript anpassen.

Für den vollen Trainings-Workflow (falls ein Modell neu trainiert werden
soll): siehe dynamics/training/ (nummerierte Pipeline) oder
dynamics/msnn/src/ (Felix' Originalcode + unsere Experimente).
