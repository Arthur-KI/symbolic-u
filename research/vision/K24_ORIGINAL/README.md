# K24 Vision Curriculum — symbolic, non-neural

## Image rules
- 512x512 PNG
- no text, shadows, gradients, outlines or textures
- objects do not touch
- at least ~25 px between objects and ~30 px from border

## Backgrounds
- white = RGB (255,255,255)
- lightgray = RGB (225,225,225)

## Object colors
- red = RGB (220,40,40)
- blue = RGB (40,80,220)
- green = RGB (40,180,70)

## Shapes
- triangle = filled triangle, clearly 3 corners
- quadrilateral = filled square or simple rectangle, clearly 4 corners

## Counts
Use exactly the count in curriculum.csv. All objects in one training image have the same color and shape.

## Anti-cheating
Training supervision lives in curriculum.csv. The visual extractor NEVER parses filenames. Held-out files must be named only test_01.png ... test_06.png.

## Allowed classical visual observations
- connected region
- mean RGB
- contour
- raw corner count
- area / centroid
- region count

The sensor must NOT emit semantic labels such as red, blue, green, triangle, quadrilateral. Those mappings are learned by U from curriculum.

## Target queries
- background?
- object color?
- object shape?
- how many objects?
- derive shape from corner structure
- derive count from regions / arithmetic
- keep color/shape/count separated on unseen combinations
- ambiguous evidence -> KEY 0
- rejected visual U does not imply KEY -1
- query remains read-only
