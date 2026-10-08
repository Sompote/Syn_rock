# Notes for Claude

This repository generates synthetic tunnel faces in Rhino 8 and labels them in the ROCK MASS LABELING COCO schema. Read README.md first. The Rhino side (`scripts/rhino_tunnel_face.py`) is IronPython 2.7 and runs inside Rhino, so keep it free of Python 3 syntax such as f-strings, and test changes on one seed before a batch.

- Rhino is reached through the RhinoMCP plugin on 127.0.0.1:1999 (`mcpstart` in Rhino). Batches use `scripts/rhino_bridge.py`; the `mcp__rhino__*` tools are for looking at the scene (`capture_viewport`) and small edits.
- The generator refuses a document with untagged objects. Never delete or overwrite the user's own Rhino geometry; open a new document with `_-New _None` instead.
- `ViewCapture` drops vertex colours when shadows are on; keep `view.CaptureToBitmap(size, mode)`. Close files explicitly in IronPython or `json.dump` writes nothing.
- Do not kill a batch while Rhino is rendering a seed. Killing the bridge client mid-request left the RhinoMCP plugin unresponsive to every client until the user ran `mcpstop` / `mcpstart` in Rhino. Let the current seed finish, then stop.
- One job at a time: the Rhino document and `output/job.json` are shared, so a second run overwrites the first mid-render. Editing `rhino_tunnel_face.py` while a batch runs changes the remaining seeds.
- The RMC codebook is not defined in the source data. The `CLASSES` table in `batch_tunnel_faces.py` is an assumed A (best) to E (worst) ordering; say so whenever class meaning matters, and do not present synthetic labels as geologically rated.
- After any visual change, look at the overlay or a contact sheet (`scripts/contact_sheet_tunnel_faces.py`) before reporting success. `fbm()` has a typical amplitude of only about +-0.3, so noise amplitudes in job parameters look smaller than their names suggest.
