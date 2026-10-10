# Geological patterns for synthetic tunnel faces

This note fixes what the generator must vary so that a model pretrained on synthetic faces sees the range of conditions a tunnel geologist maps at the face, and it fixes the label scheme used for pretraining. It draws on the ISRM suggested methods for describing discontinuities and weathering (ISRM 1978, 1981), the GSI chart for jointed and heterogeneous rock masses (Hoek and Marinos 2000; Marinos and Hoek 2001; Marinos, Marinos and Hoek 2005) and common face-mapping practice in drill-and-blast and TBM tunnels.

## Pretraining labels: GSI structure classes

The ROCK MASS LABELING codes (RMCA01 to RMCE03) have no codebook and depend on the site and the labeller, so they are a poor target for a foundation model. The pretraining target is instead the structure axis of the GSI chart, which describes the block geometry of a rock mass independently of the site. Each face zone is drawn from one structure class, and its parameters follow from that class. The RMC labels are still written as a second label set through an assumed mapping (`CLASSES` in `scripts/batch_tunnel_faces.py`), and they should be treated as provisional.

| id | GSI structure class | Generator definition |
|---|---|---|
| 0 | Intact or massive | at most one persistent set, or all spacings above 1.5 m; low relief, smooth blast facets |
| 1 | Blocky | three sets, spacing 0.6-1.5 m, well-defined cubical blocks |
| 2 | Very blocky | four or more sets or spacing 0.2-0.6 m, angular interlocked blocks |
| 3 | Blocky, disturbed, seamy | folded or faulted, many intersecting sets with persistent bedding or schistosity, spacing 0.1-0.3 m |
| 4 | Disintegrated | heavily broken, crushed rock, spacing below 0.1 m, mixture of angular and rounded pieces |
| 5 | Laminated or sheared | closely spaced persistent schistosity, lamination or shear planes, little or no blockiness |
| 6 | Shotcrete | sprayed concrete lining and support at the face perimeter |

Muck, the floor, people, equipment and background stay unlabelled, as in the real data.

## Patterns the generator varies

**Discontinuity sets.** The number of sets (1 to 4 plus random fractures), the trace orientation on the face, the mean spacing (ISRM classes from extremely close, below 20 mm, to very wide, 2-6 m), irregular spacing, persistence (traces that stop within rock bridges), aperture shown as dark traces and infilling shown as staining or light mineral lines.

**Bedding, lamination and foliation.** A dominant persistent set with layer-to-layer colour banding (sandstone-shale alternation, laminated shale, gneissic banding, schistosity). It can be folded with a set wavelength and amplitude, which gives the disturbed and laminated classes their typical look.

**Faults and shear zones.** Bands of weaker rock bounded by irregular, stepped traces with a clay gouge seam, as already implemented. Contacts between rock types and dykes (a band of a different, usually darker and fresher, rock with sharp walls) use the same zone mechanism.

**Veins.** Thin light quartz or calcite veins, single or in sets, sometimes cross-cutting joints.

**Weathering.** ISRM grades I (fresh) to V (completely weathered). Grades II-III add iron staining along joints and discoloured block rims, and grades IV-V turn the surface into a smooth, lumpy, soil-like mass with only relict structure, which is how the disintegrated class looks when it is also weathered.

**Karst and cavities.** Solution cavities in limestone, shown as dark recesses with rounded rims.

**Groundwater.** Dry, damp, wet, dripping and flowing conditions, shown as dark wet patches, glossy wet surfaces and dark streaks running down from joints and the gouge seam.

**Excavation marks.** Half-barrels of perimeter blast holes on the walls, overbreak and blast-fractured facets on the face.

**Support.** Shotcrete with a ragged front edge, rock-bolt plates and wire mesh on the lining, and steel arches.

**Rock types.** Granite and other crystalline rock (coarse speckle), sandstone (tan, bedded), shale and mudstone (dark, fissile), limestone (light grey, calcite veins, karst), schist and gneiss (foliated, banded, slight sheen) and basalt (dark, fine-grained).

**Photography.** Full-face views and close-ups that fill the frame (most real images are close-ups or crops), camera tilt and lens choice, flash at the camera or work lights from the side, colour temperature, exposure, scale bars and other occluders such as ventilation ducts and cables.

## Known gaps

TBM faces seen through the tail-shield window (circular segment lining, cutter grooves), which make up a large share of the real class-A images, are not modelled yet. The muck pile uses random convex fragments and does not settle physically. The colour of each rock type comes from fixed palettes plus the real colour pool, so unusual lithologies are under-represented.
