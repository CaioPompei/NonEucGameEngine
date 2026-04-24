# NonEucGameEngine

A Python-based OpenGL learning engine focused on first-person camera movement, shader-driven rendering, and modular engine structure.

## Current Project Status

The project is in an **early prototype stage**.

### Implemented

- OpenGL 3.3 core-profile window creation via GLFW
- Render loop with depth testing enabled
- Shader compilation/linking pipeline (vertex + fragment shaders)
- Basic mesh upload and rendering using VAO/VBO
- Real-time camera with:
  - Mouse look (yaw/pitch)
  - WASD movement
  - Vertical movement (Q/E)
  - Frame-rate independent movement using `delta_time`
- Perspective and view matrix updates every frame
- Single 3D cube rendered in the scene

### Not Implemented Yet

- Scene graph / entity system
- Multiple mesh loading (OBJ/GLTF)
- Textures and materials
- Lighting system
- Physics/collision
- UI/debug overlay
- Build packaging and automated tests

## Controls

- `W` / `A` / `S` / `D`: Move
- `Q` / `E`: Up / Down
- Mouse: Look around
- `ESC`: Exit

## Tech Stack

- Python
- PyOpenGL
- GLFW (Python bindings)
- NumPy
- Pyrr

## Requirements

- Python 3.10+
- GPU/driver support for OpenGL 3.3+
- On Windows, up-to-date graphics drivers are recommended

## Setup

1. Create and activate a virtual environment (recommended):

   **Windows PowerShell**

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```bash
   pip install glfw PyOpenGL numpy pyrr
   ```

## How To Run

From the project root:

```bash
python main.py
```

If everything is configured correctly, a window titled **"What is There?"** opens and displays a 3D cube.

## Project Layout

```text
NonEucGameEngine/
├─ main.py
└─ core/
   ├─ __init__.py
   ├─ window.py
   ├─ camera.py
   ├─ shader.py
   └─ mesh.py
```

## Module Responsibilities

- `core/window.py`: GLFW window/context lifecycle and frame operations
- `core/camera.py`: Camera transforms and keyboard/mouse input logic
- `core/shader.py`: GLSL compile/link and uniform upload helpers
- `core/mesh.py`: GPU buffer setup and geometry draw call
- `main.py`: Engine bootstrap, render loop, and scene setup

## Known Notes

- Mouse is captured on startup (`CURSOR_DISABLED`) for FPS-style control.
- This prototype currently uses hardcoded cube vertex data in `main.py`.

## Suggested Next Milestones

1. Introduce a reusable `Transform` + entity abstraction.
2. Add indexed meshes (EBO) and model loading.
3. Add basic directional light + normals.
4. Add a `requirements.txt` and simple smoke test for startup.
