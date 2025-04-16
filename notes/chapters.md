# Gliederung Masterarbeit

<!---------------------------------------------->

## Einleitung

### Projektübersicht

- Dart Scoring
  - Nutzen des Scorings
  - Scoring durch Single-Camera-System
- Verwendung von KI zur Vorhersage von Dartpfeil-Positionen
  - Was für ein Machine Learning-Problem?
- DeepDarts-System
  - Herangehensweise
    - YOLO-Netzwerk
    - ...
  - Ergebnisse
    - ...
  - Schwachstellen
    - stark limitierte Datenlage bzgl. Diversität
    - eher Proof-of-Concept als einsetzbares System
    - keine Generalisierbarkeit durch Overfitting

### Warum synthetische Datenerstellung?

- KI-Training basiert auf Daten
- Korrektheit von Daten relevant
- Generierung von Daten als Mittel zur Erstellung ausreichender Menge
- viele Daten erstellen mit wenig Aufwand
- Umfang der Daten klar definiert: Dartpfeile auf Dartscheibe verteilen
  - schematische Beschreibung der Daten möglich
  - Generierung von Trainingsdaten für KI nicht unüblich: synth_data_*

### Related Work

- Prozedurale Datenerstellung für Spiele:
  - proc_data_games[1,2,3]
  - Erstellung zufälliger Spielumgebungen auf Grundlage zufälliger Generation
  - hier: zufällige Generierung von Darts-Runden statt Welten
  - Konzept gleich, wird bereits genutzt
  - PGD-G: Procedural Data Generation for Games

- Dart-Scoring-Systems
  - GitHub-Projekte
    - darts_project[1,2,3,4]
    - Dart-scoring mittels Webcams + vorherigem Setup
  - Multi-Cam system
    - 5 Kameras
    - dart_scoring_multicam
  - Mikrofon-System
    - akustische Triangulierung
    - dart_scoring_microphone

### Aufbau der Arbeit

- Aufbau der Arbeit beschreiben
- Start: Datengenerierung
- Danach: CV-Verarbeitung
  - Entzerrung / Normalisierung
- Danach: KI-Training
  - Dartpfeil-Erkennung
- Jeweils: Grundlagen, Methodik, Implementierung, Ergebnisse
- Grund für Aufbau:
  - Projekte weitestgehend voneinander abgekapselt
  - Schnittstellen der Systeme klar definiert
  - Komplexes Projekt, Betrachtung der einzelnen Systeme zur Übersicht
  - Thematische Kapselung der Arbeit
- Danach: Diskussion der jeweiligen Systeme
- Zuletzt: Fazit der Arbeit + Future Work

<!---------------------------------------------->

## Datenerstellung

### Grundlagen (Datenerstellung)

#### Datenerstellung-Basics

- Prozedurale Datenerstellung
  - Was ist es?
  - Wozu ist es gut?
- Rendering generell
  - Ray Tracing (kurz erklären)
- Funktion einer Kamera
  - Brennweite
  - Öffnungswinkel
  - ISO
  - Film Grain / Lens Distortion
- Masken
  - Was bringt die Erstellung von Masken?
    -> Informationen über Szene extrahieren

#### Darts

- Dartscheiben-Geometrie
  - Aufbau, Aussehen, Toleranzen
- Dart-Terminologie
  - Tip, Shaft, Barrel, Flight
  - Spinne

#### Texturierung

- Material und Texturen
  - Shader
  - Licht-Eigenschaften -> Reflektivität / Roughness
  - Normal Maps
- Noise-Texturen
  - Arten von Noise
    - White Noise
    - Perlin Noise
  - Seeding
  - Thresholding / Maskierung
    -> warum braucht man das? Wozu wird es genutzt?
    - Bezug: Thresholding von Texturen untereinander
- Prozedurale Texturen
  - Warum ist es wichtig? -> Parametrisierung

---

---

### Methodik (Datenerstellung)

- 3D-Szene -> was ist überhaupt notwendig?
  - Dartscheibe
  - Pfeile
  - Lichter
  - Kamera
  - Parametrisierung
    -> Prozedurale / randomisierte Implementierung

- Scripting:
  - Wozu externes Skript?
  - Einfluss der Parametrisierung
    - Alter + Einfluss / Umsetzung
    - Welche Bereiche der Texturen werden damit beeinflusst?
  - Manuelle Beeinflussung der Szene
    - Texte auf Dartscheibe
  - Statische Objekte vs. dynamische Objekte
    - Welche Objekte werden angezeigt, welche nicht?

- Material + Licht
  - Dartscheibe - Material
    - grob beschreiben, WAS simuliert ist, aber NICHT WIE
  - Dartpfeile - Zusammensetzung
    - erklären, DASS sie zusammengesetzt sind, nicht WIE
  - Unterschiedliche Lichter + ihre Daseinsberechtigungen
    - auch Environment-Texturen -> Beleuchtung
  - Details dazu in Implementierung -> Verweis

- Post-Processing
  - Entzerrung anhand von Orientierungspunkten
    -> Homographie-Erstellung
  - Dartpfeil-Positionen in Render + entzerrtem Bild
  - Statistiken erheben (Dartpfeil-IoU, Anzahl verdeckte Tips, Board-Geometrie)

---

---

### Implementierung (Datenerstellung)

- Dartscheiben-Parametrisierung
  - Aussehen / Nutzung von Noise-Texturen -> WIE wurde es umgesetzt?
  - Variation der Spinne anhand von Noise-Verschiebung, anhängig von Alter
- Dartpfeil-Zusammensetzung
  - WIE sind die Dartpfeile zusammengesetzt?
  - Nutzung von Geometry-Nodes

- Render-Einstellungen
  - Farbexport realistisch + wie Handykamera

- Generierung von Positionen
  - Heatmaps
  - \+ Scoring
- Ermittlung von Kameraparametern
  - Brennweite in Abhängigkeit von Abstand
  - Auflösung in Abhängigkeit von Seitenverhältnis
  - Fokuspunkt um Dartscheibe
- Berechnung von Entzerrung
  - Orientierungs-Masken -> Punkte -> Homographie

---

---

### Ergebnisse (Datenerstellung)

- Beispiel-Render darstellen
  - Variationen aufzeigen
- Erstellungszeit ~30s/Sample
  - Erstellen von Daten headless auf GPU-Server möglich
- sichtlicher (qualitativer) Unterschied zwischen echten Aufnahmen und gerenderten Aufnahmen
  - augenscheinlich kein Fotorealismus in den Rendern
  - Unterscheidung zwischen echten und gerenderten Aufnahmen möglich
  - ...aber nah dran
  - Gründe dafür finden und aufzählen!
    - Shader-Komplexität
    - Scans von echten Dartscheiben / Erweiterung der prozeduralen Texturen
    - PBR (Physically-based rendering)
- Korrekte Annotation von Dartpfeilen
- Entzerrung mit (augenscheinlich) minimalen Verschiebungen
  - Grundlage: Masken-Bilder der Orientierungspunkte
  - ungenaue Punkterkennung
    -> durch Geometrie-Differenzen
    -> durch Kameraperspektive

<!---------------------------------------------->

## CV

### Grundlagen (CV)

- Polarlinien
- Binning
  - soft-binning vs hard-binning
- Thresholding
- Filterung
  - Sobel

- Kantenerkennung
- Harris Corner Detection
- Hough-Transformation
- Affine / Projektive Transformations-Matrizen
  - Rotation
  - Skalierung
  - Translation
  - Scherung

- Logpolare Entzerrung

- Farbräume
  - HSV
  - YCbCr
  - Lab
- SSIM

### Methodik (CV)

- Warum CV-Algorithmus und nicht KI?

- Preprocessing
  - Skalierung

- Kantenerkennung
  - Filterung
    - Kontrasterhöhung + Weichzeichnung
    - Annahmen für 15x15-Filter
  - Skelettierung

- Linienerkennung
  - Linienfindung
    - Hough-Lines
  - Mittelpunktextraktion
    - Winkel-Binning
  - Linienfilterung
    - Abstands-Berechnung
  - Winkelentzerrung: alle Feld-Winkel gleich
    - Rotation erster Linie
    - Scherung der Orthogonalen
    - vertikale Skalierung mit minimalem Fehler
    - Wiederholung für alle Linien als Start

- Orientierung
  - Orientierungspunkte finden
    - Logpolare Transformation
    - Corner Detection
      - Farbraum-Transformation
        - CrYV-Farbraum erklären
      - Surroundings-Thresholding (Ecken klassifizieren schwarz/weiß/farbig)
  - Orientierungspunkte klassifizieren
    - Abgleich Mean-Surrounding
      - Lab-Farbraum
      - SSIM-Abgleich
      -> Ringe: Innen-/Außenseite
    - Logpolare Position zu Winkel umwandeln
  - Homographiefindung
    - Thresholding: Positionen + Orientierung -> Klassifizierung in Ringe
    - Berechnung von Ziel-Positionen anhand von Winkel + Entfernung
  - Entzerrung
    - Nicht alle Orientierungspunkte nutzen, da Outlier dabei sein können
    - RANSAC-Herangehensweise -> 75% aller Punkte + 16 Durchläufe
    - Median-Homographie aller Entzerrungen

### Implementierung (CV)

- Winkelfindung aus gefilterten Linien
  - Adaption der Hough-Transformation
  - Codebeispiel: get_rough_line_angles(...)
- evtl. Winkelentzerrung: undistort_by_lines(...)
  - Verwendung von Matrizen und Transformationen + Auswirkungen auf Winkel
- Farbidentifizierung mittels CrYV
  - is_black(...)
  - is_white(...)
  - is_color(...)
- Surroundings-Klassifizierung: extract_surroundings(...)
  - top/bottom left/right black/white/color
  - Kombination der Ecken -> Art der Surrounding (innen / außen von Ring)
  - Abgleich mit mittlerem Surrounding

### Ergebnisse (CV)

- Metriken erklären
  - Erfolgreiche Identifizierungen
  - Mittlere Orientierungspunkt-Distanz der Transformation
  - Pixel-Distanz zu Vorhersagen

- Auswertung:
  - Statistik mit Paper-Daten
    - MA-System vs DD-System
  - Statistik mit Render-Daten
    - MA-System vs DD-System
  - ggf. Statistik mit echten Daten
  - Dauer / Geschwindigkeit
    - Schwer zu messen, da DD-System alles in einem Schritt macht
  - Fehlerursachen
    - Analyse über Metadaten der Render-Daten

- Auswertungen:
  - Simulations-Daten, Deepdarts-Daten
  - Systeme: Eigenes System, DeepDarts-d1, DeepDarts-d2
  - Metriken:
    - Geschwindigkeit: DeepDarts besser als MA
    - Genauigkeit: MA VIEL besser als DeepDarts auf nicht-DD-Daten

<!---------------------------------------------->

## KI

### Grundlagen (KI)

- Neuronale Netze
  - Grundlagen
  - CNNs
  - Klassifizierung + Regression

- Terminologie
  - Trainingsdaten
  - Validierungsdaten
  - Testdaten
  - Out-of-distribution-training
  - Over-/Underfitting
  - Loss-Funktionen
    - Overfitting

- Training / Backpropagation
  - Optimizer

- Augmentierung
  - Pixel-Augmentierung
  - Transformations-Augmentierung
  - das evtl. in Methodik?

- YOLOv8
  - Anwendungsbereich
    - Echtzeitanwendung
    - Objekterkennung
  - Multi-Scale-Output
  - Bounding Boxes
  - Non-maximum-suppression (?)
- ~Subclassing / Vererbung / Objektorientierung~

- Oversampling der Daten
  - künstliches Hinzufügen von Daten, die selten gesehen werden
  - Klassen "rot" und "grün" wurden kaum vorhergesagt
  - 20480x uniform verteilt
  - 4096x Ringe spezifisch

### Methodik (KI)

- YOLOv8
  - Warum dieses Modell?
  - Adaption des Modells
    - Multi-Scale-Output nicht verwenden
    - keine Bounding Boxes
    - Dreiteilung der Outputs:
      - Existenz
      - Position
      - Klasse
  - Konfiguration des Netzwerks
    - YOLOv8-n bzw. YOLOv8-x
    - Konfiguration erklären -> woher kommen die Varianten?
    - Anzahl Parameter
    - Vergleich mit DeepDarts-Netzwerk

- Loss-Funktionen:
  - Welche Hintergründe / Zielsetzung
  - Zusammensetzung mehrerer Losses
    - Existenz, Klassen, Positionen
    - Gewichtung der Losses
  - Warum nicht DIoU?
    - Einbindung von Klassen schwierig
    - ansonsten ähnlich zu Positions-Loss
    - Komzept ambivalent/komplex; Mehrwert im Training war nicht sichtbar

- Training
  - Welches Ziel? Wie wurde trainiert?
    - welcher Optimizer?
    - Out-of-distribution-Training
  - Trainingsdaten
    - Quellen
      - Generierte Daten + Salting mit echten Daten
      - Gewichtung unterschiedlicher Quellen
    - Augmentierung
    - Art der Outputs
      - Wie werden dem Netzwerk die Daten präsentiert?
  - Validierungsdaten
    - Quellen
  - dynamisches Training:
    - adaptive Learning Rate
    - anfängliches Steigern der Learning Rate zur Stabilisierung von Anfangsfluktuationen

- Oversampling der Daten
  - künstliches Hinzufügen von Daten, die selten gesehen werden
  - Klassen "rot" und "grün" wurden kaum vorhergesagt
  - 20480x uniform verteilt
  - 4096x Ringe spezifisch

### Implementierung (KI)

- YOLOv8: TensorFlow statt PyTorch
  - eigene Implementierung
  - Grundlage: YOLOv8-Docs (eigentlich vom Bild, aber das stammt von Config-Datei aus Repo)
  - Layer-Subclassing

- Training
  - Rahmenbedingungen
    - GPUs: NVIDIA GeForce RTX 4090
    - Batch Size
    - Learning Rate
    - Epochen
    - Datenmengen (Training + Validierung)
    - Augmentierugsparameter

  - Batch-Sizes, Datenmenge, Epochen, Augmentierungsparameter, Learning Rate
  - tf.data-Pipeline
    - Optimierung durch Auslagerung von Berechnungen auf GPU
    - Paralleles Laden von Daten

### Ergebnisse (KI)

- Metriken:
  - Anzahl korrekter Dartpfeile identifiziert
  - Anteil korrekte Felder identifiziert
  - PCS (DeepDarts-Metrik)
    - als Vergleich zur System-Performance
  - Abstand korrekter Positionen
- Vergleich DeepDarts:
  - DeepDarts-Daten
  - Rendering-Daten (Validation / Test, aber nicht Training!)
  - Echte Daten
- Vergleich unterschiedlicher Datenquellen
  - Inferenz auf generierten Daten vs. DD-Daten vs. echte Daten (nicht aus Validierungs-Set)
  - laut Trainingsanalysen ist Inferenz auf generierten Daten besser als auf anderen

<!---------------------------------------------->

## Diskussion

### Datenerstellung (Diskussion)

- nicht fotorealistisch, aber für Training mit KI reicht es
  - Grundkonzept ist klar
  - out-of-distribution-Training funktioniert (wie gut?)
  - Augmentierung so stark, dass Unterschied zwischen echten und generierten Daten verschwimmt (sollte man denken)
- Anzahl unterschiedlicher Dartpfeile stark limitiert
  - Barrels + Shafts stark limitiert -> ggf. Overfitting bei KI-Training
- prozedurale Texturen vs. Scans
  - Flexibilität vs. Realismus
- Entzerrung nicht 100% genau
  -> keine *perfekten* Trainingsdaten
- Umgebungen könnten variabler sein
  - Mehr Hintergründe
  - unterschiedlichere Beleuchtungen
  - Aufnahmewinkel teilweise nicht realistisch
- Statisches Compositing sorgt für Bias
  - Könnte mit Parametern ausgestattet werden
- Umgebungen nicht realistisch
  - andere Objekte auf Dartscheibe / stark beschädigte Dartscheibe / Verzierungen bzw. Dekoration an und um Scheibe / ...
  - kein direkter Hintergrund der Dartscheibe
    - keine Reflexionen des Lichts
    - keine Umgebungsbeleuchtung, nur direkt

### CV (Diskussion)

- Dauert lange im Gegensatz zu DD-System / einfacher KI-Inferenz
  - unfairer Vergleich, da KI auf Graphen kompiliert und optimiert ist; CV ist interpretierter Python-Code
- nicht 100%, aber ganz gut, wenn es klappt
- durch RANSAC nicht deterministisch
- klappt nicht immer
  - Parameter können angepasst werden
- relativ robust gegen Outlier
- robust gegen Verdeckung der Dartscheibe
- ist keine KI
  - man weiß, wie es funktioniert
  - man kann es debuggen

### KI (Diskussion)

- kein bereits trainiertes Modell genutzt
  - YOLOv8 wurde mit PyTorch erstellt
  - eigene Expertise liegt in TensorFlow
- größeres Modell als Referenz-Paper
  - ~6M vs. ~17M Parameter
  - Aufgabe ist aber auch komplexer
    - DD-KI unterliegt massivem Daten-Bias und ist stark overfitted
- striktes Out-of-distribution-Training möglicherweise nicht optimal
  - sichtbarer Unterschied in unterschiedlichen Quellen aus Validierungsdaten
  - generierte Validierungsdaten deutlich besser erkannt als echte Aufnahmen
  - sichtbare Schwachpunkte von Out-of-distribution-Training

<!---------------------------------------------->

## Fazit

- es ist sehr schwer, alle Gegebenheiten in automatisierter Datenerstellung zu erfassen
  - Training auf echten Daten möglicherweise notwendig
- erstellte Daten sind ausreichend, um ein neuronales Netz grundlegend auf Datenlage zu trainieren
  - Inferenz nicht fehlerfrei möglich
  - aber trotzdem zu weiten Teilen gute Ergebnisse
  - Ausmerzung von Feinheiten notwendig, um zuverlässig zu funktionieren
- im Gegensatz zu DD-System nicht overfitted
  - Inferenz auf neuen Daten grundlegend möglich
- robustes Entzerren von Dartscheiben möglich
  - kein Training notwendig
  - orientiert an grundlegenden Gegebenheiten, nicht an Daten
  - klar strukturiert, Fehlerquellen präzise zu erkennen
  - gutes System! *pat pat*

## Ausblick

### Datenerstellung (Ausblick)

- Implementierung von PBR in Datenerstellung
- Verbesserung der Datengenerierung, um realistischer zu werden und mehr Umgebungsbedingungen zu simulieren
- Datenerstellung auf weitere Farben und Formen der Dartscheibe erweitern
  - z.B. blau-rote Felder
  - ist aber meist nicht in Steeldarts gegeben, sondern eher in elektronischen Dartscheiben
    - und bei elektronischen Dartscheiben ist dieses System ohnehin überflüssig

### CV (Ausblick)

- Kompilierung der CV-Pipeline
  - entweder Cython / Numba oder Implementierung in kompilierter Sprache
- Ellipsen-Erkennung in CV einbauen (\cite{ellipse_detection_algorithm})

### KI (Ausblick)

- neues Trainieren des Systems auf mehr echten Daten
- Quantisierung der Netzwerke
- KI-Prediction auf Grundlage einer leeren Dartscheibe
  - Kalibrierungs-Bild schließen und als Referenz nutzen
  - Wenn bekannt ist, dass keine Dartpfeile auf Kalibrierungs-Bild vorhanden sind, ist die Wahrscheinlichkeit von Fehlklassifikationen des Hintergrundes geringer
- Warm Starts der Learning Rate: lr_warm_restart
