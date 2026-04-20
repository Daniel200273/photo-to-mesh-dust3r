import os
import sys
import subprocess
from pathlib import Path

# Ensure 'rich' is installed for the beautiful UI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
except ImportError:
    print("Installing 'rich' for beautiful CLI UI...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint

console = Console()

def run_step(cmd, message):
    """Run a subprocess with a gorgeous loading spinner that streams output dynamically."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(message, total=None)
        
        # Run command and capture output dynamically
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read lines in real-time. universal_newlines handles \r from tqdm bars!
        for line in process.stdout:
            clean_line = line.strip()
            if clean_line:
                # Strip excessive length to prevent terminal wrapping mess
                short_line = clean_line[:70] + "..." if len(clean_line) > 70 else clean_line
                progress.update(task, description=f"{message}  [dim]{short_line}[/dim]")
                
        process.wait()
        
        if process.returncode != 0:
            console.print(f"\n[bold red]❌ Error during: {message}[/bold red]")
            console.print("Please check the terminal output above for the exact error trace.")
            sys.exit(1)
            
        progress.update(task, completed=100, description=f"[bold green]✔ {message}[/bold green]")

def show_help():
    help_text = """[bold cyan]Available Filters & Parameters:[/bold cyan]

[bold yellow]1. Low Poly[/bold yellow] -> Decimates the mesh into sharp triangles.
    [dim]Parameter:[/dim] Target Triangles (recommended: 50-1000). Lower = blockier. Holes auto-closed.

[bold yellow]2. Voxel[/bold yellow] -> Minecraft-style blocks.
    [dim]Parameter:[/dim] Voxel Size (recommended: 0.003). Axes align to pedestal plane.

[bold yellow]3. Soft Voxel[/bold yellow] -> Voxels with smooth, melted edges.
    [dim]Parameter:[/dim] Voxel Size (recommended: 0.003). Axes align to pedestal plane.

[bold yellow]4. Hologram[/bold yellow] -> Sci-fi glowing wireframe.
   [dim]Parameter:[/dim] Optional RGB color. Leave blank for default blue hologram.

[bold yellow]5. FF7 (Retro)[/bold yellow] -> Final Fantasy 7 PS1 style (sharp polys + quantised colors).
    [dim]Parameter:[/dim] Target Triangles (recommended 300-1000). Holes auto-closed.

[bold yellow]6. Material[/bold yellow] -> PBR material injection (e.g. glass, plastic, clay).
    [dim]Parameter:[/dim] Texture image path + shininess + scale.
"""
    
    console.print(Panel(help_text, title="🎨 Stylization Guide", border_style="blue"))


def main():
    os.system('clear' if os.name == 'posix' else 'cls')
    
    console.print(Panel.fit(
        "[bold magenta]✨ FewShot-NeRF Studio Wizard ✨[/bold magenta]\n"
        "[dim]The simple, interactive 3D reconstruction pipeline.[/dim]",
        border_style="magenta"
    ))
    
    # ── 1. Preparation ───────────────────────────────────────────────────────
    raw_dir = Path("data/raw_data")
    console.print(f"\n[bold]1. Image Loading[/bold]")
    console.print(f"Please place your object photos in: [bold cyan]{raw_dir.absolute()}[/bold cyan]")
    
    Prompt.ask("[green]Press Enter when you are ready to begin[/green]")
    
    if not raw_dir.exists() or not any(f.suffix.lower() in {'.jpg', '.png', '.jpeg', '.heic'} for f in raw_dir.iterdir() if f.is_file()):
        console.print(f"[bold red]Error: No valid images found in {raw_dir}![/bold red]")
        sys.exit(1)
        
    python_exe = sys.executable

    mesh_path = "data/processed_data/mesh/object_mesh.ply"
    pedestal_path = "data/processed_data/mesh/pedestal.ply"
    final_path = "data/processed_data/mesh/final_mesh.ply"
    current_bg = "studio_cool"

    viewer_choice = Prompt.ask(
        "\n[bold]Viewer Mode[/bold] -> [bold cyan]L[/bold cyan]ocal Open3D or [bold cyan]W[/bold cyan]eb browser?",
        choices=["L", "W", "l", "w"],
        default="L",
    )
    viewer_mode = "web" if viewer_choice.lower() == "w" else "local"
    viewer_script = "src/visualize_mesh_browser.py" if viewer_mode == "web" else "src/visualize_mesh.py"
    console.print(
        "[dim]  → Using web-browser previews.[/dim]"
        if viewer_mode == "web"
        else "[dim]  → Using native Open3D previews.[/dim]"
    )

    def launch_viewer(
        mesh_file: str | Path,
        *,
        hologram_mode: bool = False,
        unlit_mode: bool = False,
        shininess: float | None = None,
        bg: str = "studio_cool",
    ) -> None:
        cmd = [
            python_exe,
            viewer_script,
            "--path",
            str(mesh_file),
            "--bg",
            bg,
        ]

        if Path(pedestal_path).exists():
            cmd.extend(["--pedestal", pedestal_path])
        if hologram_mode:
            cmd.append("--hologram")
        if unlit_mode:
            cmd.append("--unlit")
        if shininess is not None:
            cmd.extend(["--shininess", str(shininess)])
        if viewer_mode == "web":
            cmd.append("--open")

        subprocess.run(cmd)

    # ── 2. Run Pipeline (or Skip) ────────────────────────────────────────────
    skip_pipeline = False
    if Path(final_path).exists():
        console.print(f"\n[cyan]ℹ️  Found an existing 3D reconstruction ({Path(final_path).name})[/cyan]")
        choice = Prompt.ask(
            "[bold]Do you want to use the existing mesh or rebuild from scratch?[/bold] [[U]se existing / [R]ebuild]",
            default="U"
        )
        if choice.upper() == "U":
            skip_pipeline = True

    if not skip_pipeline:
        console.print("\n[bold]⚙️  2. Executing Heavy Pipeline...[/bold]")
        run_step([python_exe, "src/pipeline.py"], "[bold blue]DUSt3R 3D Reconstruction[/bold blue]")
        run_step([python_exe, "src/mesh_reconstruction.py"], "[bold blue]Meshing & Surface Generation[/bold blue]")
    else:
        console.print("\n[bold]⏭️  Skipping Heavy Pipeline... Using Existing Mesh![/bold]")
    
    # ── 3. View Raw Mesh ─────────────────────────────────────────────────────
    console.print("\n[bold green]🎉 Base 3D Model Successfully Created![/bold green]")
    Prompt.ask("\n[bold white]Press Enter[/bold white] to open the 3D Viewer and inspect your raw object...")

    if viewer_mode == "web":
        console.print("[dim]Opening web viewer in your browser...[/dim]")
    else:
        console.print("[dim]Opening native Open3D viewer (close the window to continue)...[/dim]")
    launch_viewer(final_path, bg=current_bg)

    # ── 4. Stylization Loop ──────────────────────────────────────────────────
    # Tracks what mesh each filter has previously produced (filter_name -> path).
    # When the same filter is re-used, its old output is silently replaced.
    filter_outputs: dict[str, Path] = {}
    # The active source mesh for the next filter (starts as the pure object mesh)
    current_source: Path = Path(mesh_path)
    # A history stack of sources to allow 'Undo' functionality
    history_stack: list[Path] = [current_source]

    Path("data/stylized").mkdir(exist_ok=True)

    filter_map = {
        "1": ("low_poly",   "Target Triangles (recommended 50-1000)",          "500"),
        "2": ("voxel",      "Voxel Size in meters (recommended 0.003)",         "0.003"),
        "3": ("soft_voxel", "Voxel Size in meters (recommended 0.003)",         "0.003"),
        "4": ("hologram",   "Optional RGB (R,G,B). Leave blank for default blue", ""),
        "5": ("ff7",        "Target Triangles (recommended 300-1000)",         "800"),
        "6": ("material",   "Material",                                       ""),
    }

    last_shininess = None

    while True:
        # ── Show menu header ──────────────────────────────────────────────
        console.print("\n" + "═"*52)
        console.print("[bold]🎨  Stylization Menu[/bold]")
        console.print("  [1] Low Poly       [4] Hologram")
        console.print("  [2] Voxel          [5] FF7 Retro")
        console.print("  [3] Soft Voxel     [6] PBR Material")
        console.print("  [B] Change BG      [H] Help / Explain Filters")
        console.print("  [Q] Quit")

        # Show which mesh is currently the active source
        is_original = current_source == Path(mesh_path)
        src_label = "[green]original mesh[/green]" if is_original else f"[yellow]stylized ({current_source.stem})[/yellow]"
        console.print(f"\n  Source for next filter: {src_label}")

        # Only show the switch prompt when a stylized result actually exists
        if len(history_stack) > 1 or filter_outputs:
            switch = Prompt.ask(
                "  Switch source? ([bold cyan]O[/bold cyan]riginal / [bold cyan]K[/bold cyan]eep current / [bold cyan]U[/bold cyan]ndo (Back) / [bold cyan]filter number[/bold cyan])",
                default="K"
            )
            if switch.upper() == "O":
                current_source = Path(mesh_path)
                history_stack.append(current_source)
                console.print("[dim]  → Switched to original mesh.[/dim]")
            elif switch.upper() == "U":
                if len(history_stack) > 1:
                    history_stack.pop() # remove current state
                    current_source = history_stack[-1]
                    console.print(f"[dim]  → Undid last step; reverted to {current_source.stem}.[/dim]")
                else:
                    console.print("[dim]  → Already at original mesh. Nothing to undo.[/dim]")
            elif switch.upper() != "K":
                # User typed a filter number — use that filter's last output if available
                filt_key = switch.strip()
                if filt_key in filter_map and filter_map[filt_key][0] in filter_outputs:
                    current_source = filter_outputs[filter_map[filt_key][0]]
                    history_stack.append(current_source)
                    console.print(f"[dim]  → Switched to {current_source.stem}.[/dim]")
                else:
                    console.print("[dim]  → No saved result for that filter yet, keeping current.[/dim]")

        choice = Prompt.ask("\nSelect an option", choices=["1","2","3","4","5","6","H","h","B","b","Q","q"])
        
        if choice.lower() == 'q':
            console.print("\n[magenta]Goodbye![/magenta] 👋")
            break
        elif choice.lower() == 'h':
            show_help()
            continue
        elif choice.lower() == 'b':
            console.print("\n[bold]🎨 Available Backgrounds:[/bold]")
            console.print("  1. studio_cool   (Best all-around object presentation)")
            console.print("  2. neutral_slate (Balanced neutral contrast)")
            console.print("  3. warm_stone    (Complements cool-colored assets)")
            console.print("  4. soft_fog      (Gentle low-fatigue backdrop)")
            console.print("  5. blue_mist     (Great for hologram/cyan accents)")
            bg_sel = Prompt.ask("Choose theme [1-5]", choices=["1","2","3","4","5"], default="1")

            themes = {
                "1": "studio_cool",
                "2": "neutral_slate",
                "3": "warm_stone",
                "4": "soft_fog",
                "5": "blue_mist",
            }
            current_bg = themes[bg_sel]
            console.print(f"[dim]  → Viewer background set to {current_bg}.[/dim]")
            continue
            
        filt_name, param_desc, default_param = filter_map[choice]

        console.print(f"\n[italic]Selected: {filt_name.upper()}[/italic]")
        
        param_val = ""
        if filt_name == "hologram":
            param_val = Prompt.ask(f"Enter {param_desc}", default=default_param)
        elif filt_name != "material":
            param_val = Prompt.ask(f"Enter {param_desc}", default=default_param)

        extra_args = []
        if filt_name == "hologram":
            if param_val.strip():
                extra_args = ["--color", param_val.strip()]
            else:
                console.print("[dim]  → Using default blue hologram color.[/dim]")
            param_val = ""
        elif filt_name == "material":
            tex_dir = Path("data/textures")
            available_textures = []
            if tex_dir.exists():
                for f in sorted(tex_dir.iterdir()):
                    if f.is_file() and f.suffix.lower() in {'.jpg', '.png', '.jpeg'}:
                        available_textures.append(f)
            
            texture_choice = ""
            if available_textures:
                console.print("\n[bold]🖼️  Available Image Textures:[/bold]")
                for i, tex in enumerate(available_textures, 1):
                    console.print(f"  [{i}] {tex.stem.replace('_', ' ').title()}")
                console.print("  [0] Type custom texture path manually")
                
                sel_str = Prompt.ask("\nSelect a texture", choices=[str(i) for i in range(len(available_textures) + 1)])
                sel_idx = int(sel_str)
                if sel_idx > 0:
                    texture_choice = str(available_textures[sel_idx - 1])
                    
            if not texture_choice:
                texture_choice = Prompt.ask("Enter absolute path to Seamless Image Texture")
                
            if texture_choice.strip():
                # Using image texture
                shininess = Prompt.ask("Enter Material Shininess (e.g. 5 for rough, 50 for glossy)", default="48")
                last_shininess = float(shininess)
                scale = Prompt.ask("Enter Texture Scale Multiplier", default="1.0")
                extra_args = ["--texture", texture_choice.strip(), "--shininess", shininess, "--scale", scale]
                src_name = current_source.stem.lower()
                if "voxel" in src_name:
                    extra_args.append("--preserve-geometry")
                    console.print("[dim]  → Preserve geometry enabled for voxel-derived source.[/dim]")
            else:
                console.print("[red]Texture path required for material styling. Aborting.[/red]")
                continue

        # ── Determine output path ─────────────────────────────────────────
        # Each filter gets one named slot; re-running the same filter overwrites it.
        out_path = Path("data/stylized") / f"{filt_name}.ply"
        if out_path.exists():
            console.print(f"[dim]  → Replacing previous {filt_name} result.[/dim]")

        # ── Build and run stylize command ─────────────────────────────────
        cmd = [
            python_exe, "src/stylize.py",
            "--input",    str(current_source),
            "--filter",   filt_name,
            "--pedestal", pedestal_path,
            "--output",   str(out_path),
        ]
        if filt_name in {"low_poly", "ff7"}:
            cmd.append("--close")
        if param_val:
            cmd.extend(["--param", str(param_val)])
        cmd.extend(extra_args)

        run_step(cmd, f"Applying [magenta]{filt_name}[/magenta] to [cyan]{current_source.stem}[/cyan]...")

        # ── Save result and update current source ─────────────────────────
        if filt_name == "hologram":
            actual_out = out_path.parent / f"{out_path.stem}_body.ply"
        else:
            actual_out = out_path
            
        filter_outputs[filt_name] = actual_out
        current_source = actual_out   # next filter will default to this result
        
        # Only push to history if we actually successfully advanced to a new state
        if history_stack[-1] != current_source:
            history_stack.append(current_source)

        # ── View result ───────────────────────────────────────────────────
        if viewer_mode == "web":
            console.print("[dim]Opening web viewer in your browser...[/dim]")
        else:
            console.print("[dim]Opening native Open3D viewer (close the window to return to menu)...[/dim]")

        launch_viewer(
            actual_out,
            hologram_mode=(filt_name == "hologram"),
            unlit_mode=(filt_name == "hologram"),
            shininess=(last_shininess if filt_name == "material" and last_shininess is not None else None),
            bg=current_bg,
        )


if __name__ == "__main__":
    main()
