================================================================
reference/vehicle — Gemessene Fahrzeugparameter + Belegfotos
================================================================

vehicle_parameters.txt (Stand 2026-08-29, vollständigste Version — ersetzt
die ältere 2026-08-13-Version, die noch in dynamics/msnn/ lag und u.a.
Außenmaße, LiDAR-Spezifikationen und Sensor-Montagepositionen nicht
enthielt):

  1. Physikalisch gemessene Parameter (Masse, Radstand, CoG, Außenmaße)
  2. Aktorik & Sensorik (Lenkwinkel-, Geschwindigkeits-, ERPM-Umrechnung,
     LiDAR-Spezifikation)
  3. Von MS-NN exp9 gelernte physikalische Parameter
  4. Referenz: JETSON4 Pacejka-Parameter (anderes Fahrzeug!)
  5. Parameter für Isaac Sim / Bicycle Model
  6. Fehlende Parameter (noch nicht gemessen)
  6b. Sensor-Montagepositionen (Kamera/LiDAR/MoCap relativ zu base_link)
  6c. Kamera-Spezifikationen (Luxonis OAK-D Pro W)
  7. Datenquellen

fotos/ — Belegfotos zu den Handmessungen in Abschnitt 1:
  radstand_32.25cm.jpg          Radstand-Messung (L = 0.3225 m)
  laenge_mit_raedern_42cm.jpg   Gesamtlänge über Räder (0.42 m)
  breite_mit_reifen_26cm.jpg    Gesamtbreite über Reifen (0.26 m)
  reifenbreite_4cm.jpg          Reifenbreite (0.04 m)
  wiegung_1/2/3.png             Achslast-Messung (SkyRC Corner Weight
                                 Scale) — Vorderachse 1860g, Hinterachse
                                 1928g, siehe vehicle_parameters.txt Abschnitt 1
  fahrzeug_foto1/2.jpg          Übersichtsfotos des Fahrzeugs

Alle Maßband-Messungen sind undatiert im Original, die Wiege-Fotos sind
vom 2026-07-28 (siehe vehicle_parameters.txt Abschnitt 1 für die Quelle:
SkyRC Corner Weight Scale).
