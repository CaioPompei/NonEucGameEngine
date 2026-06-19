# NonEucGameEngine

A Python + OpenGL 3.3 game engine built to explore **non-Euclidean rendering**
via **stencil-buffer portals**: paired portals connect distinct regions of the
world, and the renderer recurses through them so portals seen *inside* a portal
also open up (up to a configurable depth).

Around that graphics core the project grew into a small but complete engine:
a Phong/Blinn-Phong lighting pipeline with cubemap shadows, textures and a sky
cubemap, a character controller with AABB collision and physical portal
traversal, and **fully data-driven levels loaded from JSON** with triggers,
puzzles and level-to-level transitions.

## Features

### Rendering
- OpenGL 3.3 core-profile window via GLFW, with an 8-bit stencil buffer
- **Paired stencil portals**:
  - Recursive rendering up to `max_depth` (portals inside portals open too)
  - Virtual-camera view through the destination portal
  - Oblique near-plane projection (Lengyel) to clip geometry between the
    virtual camera and the destination portal
- **Lighting** (`engine/light.py`): Phong/Blinn-Phong with ambient term and up
  to 10 simultaneous lights — `PointLight`, `DirectionalLight`, `SpotLight`
- **Shadows** (`engine/shadow_map.py`): omnidirectional shadow maps (depth
  cubemap, 6 faces) for point lights, with PCF filtering; **baked** per level
- **Textures** (`engine/texture.py`): Pillow loading, mipmaps, `GL_REPEAT`
  tiling via UV scale, path-keyed cache
- **Skybox** (`engine/skybox.py`): sky cubemap drawn at the far plane
- **Static batching** (`engine/static_batch.py`): merges static geometry into a
  handful of draw calls per (texture, color) — critical because the scene is
  redrawn once per portal recursion level

### Gameplay
- **Player / physics** (`game/player.py`): two modes, toggled with `V`
  - **WALK**: acceleration/friction movement, gravity, jump with coyote time +
    jump buffering, AABB wall-slide collision, and visual head-bob
  - **FREECAM**: noclip 6-DOF fly camera for debugging
- **Collision** (`game/collision.py`): static AABB world with per-axis
  move-and-slide and sub-stepping to avoid tunneling
- **Physical portal traversal**: crossing a portal reorients position, view
  direction and velocity via a pre-baked traversal matrix
- **Triggers & puzzles** (`game/trigger.py`, `game/puzzle.py`): AABB trigger
  zones fire events; a puzzle manager decides completion and the next level
- **Data-driven levels** (`game/level_loader.py`): no geometry is hardcoded —
  scene, lights, portals, triggers, puzzle, skybox and spawn all come from JSON

### UI
- Navigable text **menu** (`engine/menu.py`)
- HUD via `engine/text_overlay.py`: center crosshair (WALK), "debug mode" label
  (FREECAM), and a toggleable stats panel (fps, mode, position, yaw/pitch,
  speed, on-ground)
- Fullscreen toggle (`F11`)

## Controls

| Key | Action |
| --- | --- |
| `W` / `A` / `S` / `D` | Move |
| `Space` | Jump (WALK mode) |
| `Q` / `E` | Down / Up (FREECAM mode) |
| `Left Shift` | Sprint |
| `V` | Toggle WALK / FREECAM |
| `´` (`KEY_LEFT_BRACKET`, BR-ABNT2 layout) | Toggle stats panel |
| `F11` | Toggle fullscreen |
| Mouse | Look around |
| `ESC` | Return to menu (quits from the menu) |

In the initial menu, navigate with `↑` / `↓` (or `W` / `S`) and confirm with
`Enter` / `Space`.

## Tech Stack

| Technology | Use | Version |
| --- | --- | --- |
| Python | Language | 3.10+ |
| PyOpenGL (+ accelerate) | OpenGL bindings | 3.1.10 |
| GLFW | Window / context / input | 2.10.0 |
| NumPy | Linear algebra / arrays | 2.4.4 |
| Pyrr | Matrices and transforms | 0.10.3 |
| Pillow | Text rasterization + image loading | 12.2.0 |

OpenGL **3.3 core profile**, with the **stencil buffer** enabled.

## Setup

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From the project root:

```bash
python main.py
```

A window titled **"What is There?"** opens with the menu. Choose **Iniciar** to
load `levels/level_01.json`, then walk toward a portal and look through.

## Project Layout

```text
NonEucGameEngine/
├─ main.py                  # Bootstrap, render loop, game states (menu/playing)
├─ requirements.txt
├─ README.md
├─ engine/                  # Generic, game-agnostic engine
│  ├─ window.py             # GLFW window + GL context lifecycle, cursor, fullscreen
│  ├─ shader.py             # GLSL compile/link + cached uniform helpers
│  ├─ mesh.py               # VAO/VBO upload + draw (pos / +normal / +UV)
│  ├─ primitives.py         # Built-in geometry (cube with normals + UVs)
│  ├─ entity.py             # Transform + mesh + texture; cached model matrix; AABB
│  ├─ scene.py              # Entities + lights; draw, depth draw, shadow bake
│  ├─ camera.py             # FPS camera (yaw/pitch + WASDQE), view offset
│  ├─ light.py              # PointLight / DirectionalLight / SpotLight
│  ├─ shadow_map.py         # Omnidirectional (cubemap) shadow maps + PCF
│  ├─ texture.py            # 2D texture loading + registry/cache
│  ├─ skybox.py             # Sky cubemap
│  ├─ static_batch.py       # Static geometry batching by (texture, color)
│  ├─ menu.py               # Keyboard-navigable text menu
│  └─ text_overlay.py       # 2D textured-quad text via Pillow
├─ game/                    # Gameplay specific to this prototype
│  ├─ level.py              # Level dataclass (scene, portals, triggers, …)
│  ├─ level_loader.py       # Builds a Level from a JSON file
│  ├─ mesh_registry.py      # Resolves mesh names → shared mesh instances
│  ├─ player.py             # WALK/FREECAM, gravity, jump, head-bob, traversal
│  ├─ collision.py          # Static AABB world + move-and-slide
│  ├─ portal.py             # Portal entity (quad, transform, normal, linking)
│  ├─ portal_renderer.py    # Recursive stencil-portal renderer
│  ├─ trigger.py            # AABB trigger zones
│  └─ puzzle.py             # Puzzle state + level transition
├─ math3d/
│  └─ portal_math.py        # Virtual view, oblique near plane, traversal transform
├─ shaders/
│  ├─ phong.vert / phong.frag
│  ├─ simple.vert / simple.frag
│  └─ depth.vert / depth.frag   # shadow-map depth pass
├─ levels/                  # Data-driven levels (JSON)
│  └─ level_01.json, level_02.json, …
├─ textures/
│  ├─ walls/                # Wall/floor textures
│  └─ skyboxes/             # Sky cubemaps
└─ Docs/
   ├─ ARCHITECTURE.md       # Module responsibilities + stencil-portal algorithm
   ├─ FUNCIONALIDADES.md    # Full feature catalog + JSON level schema (PT-BR)
   └─ *.pdf                 # Academic deliverables
```

## Levels

Levels are plain JSON files in `levels/`. Every field is optional except `name`;
absent fields fall back to sensible defaults. A level describes its scene
entities, lights, portals, triggers, puzzle, optional skybox and player spawn.
See the full schema, with annotated examples, in
[Docs/FUNCIONALIDADES.md](Docs/FUNCIONALIDADES.md) (section 4).

## Documentation

- [Docs/ARCHITECTURE.md](Docs/ARCHITECTURE.md) — module-by-module
  responsibilities and the stencil-portal algorithm in detail.
- [Docs/FUNCIONALIDADES.md](Docs/FUNCIONALIDADES.md) — complete feature catalog
  (engine vs. game), the JSON level schema, the per-frame pipeline and key
  technical decisions (PT-BR).

## References

- Lengyel, E. (2005). *Oblique View Frustum Depth Projection and Clipping.*
  Journal of Game Development, 1(2), 5–16.
- Kilgard, M. J. *Improving Shadows and Reflections via the Stencil Buffer.*
  NVIDIA whitepaper.
- *LearnOpenGL* — Point Shadows / PCF.
</content>
</invoke>
