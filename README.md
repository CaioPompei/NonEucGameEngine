# NonEucGameEngine

A Python + OpenGL prototype exploring **non-Euclidean rendering** via stencil-buffer portals. Two portals open into the same room from different angles, and the renderer recurses through them so portals seen *inside* a portal also open up (up to a configurable depth).

## Features

- OpenGL 3.3 core-profile window via GLFW, with stencil buffer enabled
- Phong lighting (ambient + diffuse + specular) in a single closed room
- **Paired stencil portals** with:
  - Recursive rendering up to `max_depth`
  - Virtual-camera view through the destination portal
  - Oblique near-plane projection (Lengyel) to clip geometry between the virtual camera and the destination portal
- Player with two modes:
  - **WALK**: gravity, jump, horizontal movement on the XZ plane
  - **FREECAM**: full 6-DOF camera (toggled with `V`)
- Debug text overlay (shown in FREECAM mode)
- External GLSL files loaded from `shaders/`

## Controls

| Key | Action |
| --- | --- |
| `W` / `A` / `S` / `D` | Move |
| `Space` | Jump (WALK mode) |
| `Q` / `E` | Up / Down (FREECAM mode) |
| `Left Shift` | Sprint (2x speed) |
| `V` | Toggle WALK / FREECAM |
| Mouse | Look around |
| `ESC` | Exit |

## Tech Stack

- Python 3.10+
- PyOpenGL + PyOpenGL-accelerate
- GLFW (Python bindings)
- NumPy
- Pyrr
- Pillow (text overlay rasterization)

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

A window titled **"What is There?"** opens. Walk toward either portal and look through.

## Project Layout

```text
NonEucGameEngine/
├─ main.py                  # Bootstrap, render loop, scene assembly
├─ requirements.txt
├─ README.md
├─ engine/                  # Generic, game-agnostic engine
│  ├─ window.py             # GLFW window + GL context lifecycle
│  ├─ shader.py             # GLSL compile/link + uniform helpers
│  ├─ mesh.py               # VAO/VBO upload + draw
│  ├─ primitives.py         # Built-in geometry (cube)
│  ├─ entity.py             # Transform + mesh wrapper
│  ├─ scene.py              # Entity list + bulk draw
│  ├─ camera.py             # FPS camera (yaw/pitch + WASDQE)
│  └─ text_overlay.py       # 2D textured-quad text via Pillow
├─ game/                    # Gameplay specific to this prototype
│  ├─ level.py              # Room geometry composition
│  ├─ player.py             # WALK/FREECAM, gravity, jump
│  ├─ portal.py             # Portal entity (quad, transform, normal)
│  └─ portal_renderer.py    # Recursive stencil-portal renderer
├─ math3d/
│  └─ portal_math.py        # Virtual view + oblique near plane
├─ shaders/
│  ├─ phong.vert / phong.frag
│  └─ simple.vert / simple.frag
└─ Docs/
   ├─ ARCHITECTURE.md
   └─ *.pdf                 # Academic deliverables
```

See [Docs/ARCHITECTURE.md](Docs/ARCHITECTURE.md) for the module-by-module responsibilities and the stencil-portal algorithm.

## References

- Lengyel, E. (2005). *Oblique View Frustum Depth Projection and Clipping.* Journal of Game Development, 1(2), 5–16.
