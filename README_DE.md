# Symbolic-U — deutsche Übersicht

**Symbolic-U ist ein experimentelles, nicht-neuronales symbolisches Lern- und
Reasoning-System.** Dieses Repository ist bewusst zweigeteilt: Es enthält die
saubere aktuelle Runtime **und** die komplette vorhandene Forschungsgeschichte.

## Das Wichtigste

```text
KEY +1 = Proposition positiv beweisbar
KEY  0 = unbekannt / unentschieden
KEY -1 = explizites Gegenteil positiv beweisbar

U +1 = Ableitung bestätigt
U  0 = Ableitung offen
U -1 = Ableitung verworfen
```

Der harte Architekturvertrag lautet:

```text
U = -1  !=  KEY = -1
```

Eine verworfene Herleitung macht die Zielaussage also nicht falsch.
Widerspruch wird separat markiert und nicht mit `UNKNOWN` verwechselt.

Die Ausführung arbeitet rückwärts von der Frage:

```text
QUERY Ziel-Key
   -> welche U könnten ihn erzeugen?
   -> welche Eingang-Keys brauchen diese U?
   -> rekursiv +1 / 0 / -1 prüfen
```

Kurzform des Projekts:

> **Vorwärts erzeugt Hypothesen. Rückwärts gibt ihnen Bedeutung.**

## Was fest ist

Der Kernel soll möglichst wenig Domänenwissen enthalten:

```text
SYMBOL / IDENTITY / ORDER
KEY / U
VARIABLE / BIND
CONTEXT / PROVENANCE
OPPOSITION
TERNARY STATE
MATCH / COMPOSE / SEARCH
BACKWARD PROVE
RESOURCE / BUDGET
CYCLE / TERMINATION
```

Deutsch, Dreiecke, Stunden, GIVE-Semantik, POS-Tags oder ein bestimmter Sensor
sollen nicht im Kernel fest verdrahtet sein.

## Was gelernt wurde

In kontrollierten Curricula wurden unter anderem untersucht:

- Sprachbindung, Ereignisbedeutung und Morphologie;
- rekursive symbolische Arithmetic-U;
- Zeitrichtung, Einheiten, Intervalle und Zustandsänderungen;
- Farbe/Form oberhalb klassischer Bildsensoren;
- echtes symbolisches Zählen aus `MEMBER`/`VERTEX_OF`;
- query- und kontextabhängige Auswahl klassischer Sensorpfade;
- Kompilierung stabiler Proofgraphs zu revidierbaren `Macro-U`.

Das ist **kein RL** und kein Gradientenlernen. Kandidaten werden anhand von
Curriculum-Konsequenzen, Support/Konflikt, Identifizierbarkeit, Blindtests und
Ablationen geprüft.

## Vision

Die Bildseite ist bewusst zweistufig:

```text
Pixel
 -> klassische OpenCV/Pillow-Filter
 -> Regionen / Vertices / Messwerte
 -> Symbolic-U
 -> Zählen / Farbe / Form / Query
```

Wir behaupten nicht, dass das System rohe Pixelwahrnehmung gelernt hat.
K25 geht einen Schritt weiter und lässt mehrere klassische Filterwege antreten;
der Reasoner lernt bedingte `TRUST-U`, ohne Störungsnamen wie "dunkel" oder
"unscharf" als Trainingssignal zu bekommen.

## Ordner

- `symbolic_u/` — aktuelle, saubere Runtime
- `tests/` — integrierte Regressionstests
- `demos/` — kleine Beispiele
- `data/vision/` — K24/K25-Bildcurriculum
- `research/language/` — kuratierte Sprachmeilensteine
- `research/math/` — Arithmetic-/Programmlernen
- `research/kernel_time/` — K18-K23
- `research/vision/` — K24/K25
- `research/archive_full/` — unverändertes Vollarchiv der wiedergefundenen frühen Artefakte
- `docs/` — zusammengefasste Dokumentation

## Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

Erwartung der aktuellen sauberen Suite:

```text
Ran 15 tests
OK
```

Zum kompletten Forschungsstand siehe:

- [Experimente K1-K25](docs/EXPERIMENTS.md)
- [Trainingsmethode](docs/TRAINING_METHOD.md)
- [Architektur](docs/ARCHITECTURE.md)
- [Vision und Sensorwahl](docs/VISION_AND_SENSORS.md)
- [Prior Art](docs/PRIOR_ART.md)
- [Grenzen](docs/LIMITATIONS.md)
- [Reproduzierbarkeit](docs/REPRODUCIBILITY.md)

Lizenz: standardmäßig **Apache-2.0**. Siehe [LICENSE](LICENSE).
