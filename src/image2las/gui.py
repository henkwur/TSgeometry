"""Tkinter GUI for image2las — file/folder selection with persistent settings."""
from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .batch import convert_root_folder, discover_envi_fused_hdr_files_filtered
from .converter import ConversionConfig, convert_image_to_las

# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------

def _settings_path() -> Path:
    """Return path to the JSON settings file next to this package."""
    try:
        import platformdirs
        config_dir = Path(platformdirs.user_config_dir("image2las", appauthor=False))
    except ImportError:
        config_dir = Path.home() / ".image2las"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "settings.json"


def _load_settings() -> dict:
    path = _settings_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_settings(settings: dict) -> None:
    _settings_path().write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("image2las")
        self.resizable(True, True)
        self.minsize(580, 520)

        self._settings = _load_settings()
        self._running = False
        self._cancel_event: threading.Event | None = None
        self._is_batch_mode = False

        self._build_ui()
        self._apply_settings()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # ---- Files frame ------------------------------------------------
        files_frame = ttk.LabelFrame(self, text="Bestanden", padding=8)
        files_frame.pack(fill="x", **pad)
        files_frame.columnconfigure(1, weight=1)

        ttk.Label(files_frame, text="Invoer (bestand of rootmap):").grid(row=0, column=0, sticky="w")
        self._input_var = tk.StringVar()
        ttk.Entry(files_frame, textvariable=self._input_var).grid(
            row=0, column=1, sticky="ew", padx=(4, 4)
        )
        ttk.Button(files_frame, text="Bladeren…", command=self._browse_input).grid(
            row=0, column=2, sticky="e"
        )
        ttk.Button(files_frame, text="Map…", command=self._browse_input_folder).grid(
            row=0, column=3, sticky="e", padx=(4, 0)
        )

        ttk.Label(files_frame, text="Uitvoermap:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self._output_var = tk.StringVar()
        ttk.Entry(files_frame, textvariable=self._output_var).grid(
            row=1, column=1, sticky="ew", padx=(4, 4), pady=(6, 0)
        )
        ttk.Button(files_frame, text="Bladeren…", command=self._browse_output).grid(
            row=1, column=2, sticky="e", pady=(6, 0)
        )

        # ---- ENVI coordinates frame -------------------------------------
        envi_frame = ttk.LabelFrame(self, text="ENVI-coördinaten", padding=8)
        envi_frame.pack(fill="x", **pad)

        self._envi_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            envi_frame, text="XYZ uit ENVI-kanalen lezen", variable=self._envi_var
        ).grid(row=0, column=0, columnspan=6, sticky="w")

        labels = ["X meter", "X fractie", "Y meter", "Y fractie", "Z meter", "Z fractie"]
        defaults = [227, 228, 229, 230, 231, 232]
        self._envi_channel_vars: list[tk.IntVar] = []
        for col, (lbl, default) in enumerate(zip(labels, defaults)):
            ttk.Label(envi_frame, text=lbl).grid(row=1, column=col, padx=4, sticky="w")
            var = tk.IntVar(value=default)
            ttk.Spinbox(envi_frame, textvariable=var, from_=1, to=9999, width=6).grid(
                row=2, column=col, padx=4
            )
            self._envi_channel_vars.append(var)

        # ---- RGB frame --------------------------------------------------
        rgb_frame = ttk.LabelFrame(self, text="RGB-kleuren", padding=8)
        rgb_frame.pack(fill="x", **pad)

        self._rgb_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            rgb_frame, text="RGB-kleuren toevoegen", variable=self._rgb_var
        ).grid(row=0, column=0, columnspan=6, sticky="w")

        rgb_labels = ["Rood kanaal", "Groen kanaal", "Blauw kanaal", "Clip laag %", "Clip hoog %"]
        rgb_defaults = [93, 54, 24, 1.0, 99.5]
        rgb_steps = [1, 1, 1, 0.1, 0.1]
        self._rgb_vars: list[tk.Variable] = []
        for col, (lbl, default, step) in enumerate(zip(rgb_labels, rgb_defaults, rgb_steps)):
            ttk.Label(rgb_frame, text=lbl).grid(row=1, column=col, padx=4, sticky="w")
            if isinstance(default, float):
                var: tk.Variable = tk.DoubleVar(value=default)
                widget = ttk.Spinbox(rgb_frame, textvariable=var, from_=0.0, to=100.0,
                                     increment=step, width=7, format="%.1f")
            else:
                var = tk.IntVar(value=default)
                widget = ttk.Spinbox(rgb_frame, textvariable=var, from_=1, to=9999, width=7)
            widget.grid(row=2, column=col, padx=4)
            self._rgb_vars.append(var)

        # ---- Output options --------------------------------------------
        output_opts_frame = ttk.LabelFrame(self, text="Output-opties", padding=8)
        output_opts_frame.pack(fill="x", **pad)

        self._write_las_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            output_opts_frame,
            text="LAS-bestand schrijven (uitzetten voor snelle test)",
            variable=self._write_las_var,
        ).grid(row=0, column=0, sticky="w")

        self._process_vnir_var = tk.BooleanVar(value=True)
        self._process_swir_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            output_opts_frame,
            text="Verwerk VNIR",
            variable=self._process_vnir_var,
        ).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(
            output_opts_frame,
            text="Verwerk SWIR",
            variable=self._process_swir_var,
        ).grid(row=1, column=1, sticky="w", padx=(12, 0))

        # ---- Convert button + log ---------------------------------------
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=(8, 2))

        self._convert_btn = ttk.Button(
            btn_frame, text="Converteer", command=self._start_convert
        )
        self._convert_btn.pack(side="left")

        self._cancel_btn = ttk.Button(
            btn_frame, text="Annuleren", command=self._cancel_convert, state="disabled"
        )
        self._cancel_btn.pack(side="left", padx=(8, 0))

        self._progress = ttk.Progressbar(btn_frame, mode="indeterminate")
        self._progress.pack(side="left", fill="x", expand=True, padx=(8, 0))

        log_frame = ttk.LabelFrame(self, text="Log", padding=4)
        log_frame.pack(fill="both", expand=True, padx=8, pady=(2, 8))
        self._log = scrolledtext.ScrolledText(log_frame, height=6, state="disabled",
                                              font=("Consolas", 9))
        self._log.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _apply_settings(self) -> None:
        s = self._settings
        self._input_var.set(s.get("last_input_file", ""))
        self._output_var.set(s.get("last_output_folder", ""))
        self._envi_var.set(s.get("envi_coordinates", True))
        envi_ch = s.get("envi_channels", [227, 228, 229, 230, 231, 232])
        for var, val in zip(self._envi_channel_vars, envi_ch):
            var.set(val)
        self._rgb_var.set(s.get("use_rgb", True))
        rgb_vals = s.get("rgb_values", [93, 54, 24, 1.0, 99.5])
        for var, val in zip(self._rgb_vars, rgb_vals):
            var.set(val)
        self._write_las_var.set(s.get("write_las", True))
        self._process_vnir_var.set(s.get("process_vnir", True))
        self._process_swir_var.set(s.get("process_swir", True))

    def _collect_settings(self) -> None:
        self._settings["last_input_file"] = self._input_var.get()
        self._settings["last_output_folder"] = self._output_var.get()
        self._settings["envi_coordinates"] = self._envi_var.get()
        self._settings["envi_channels"] = [v.get() for v in self._envi_channel_vars]
        self._settings["use_rgb"] = self._rgb_var.get()
        self._settings["rgb_values"] = [v.get() for v in self._rgb_vars]
        self._settings["write_las"] = self._write_las_var.get()
        self._settings["process_vnir"] = self._process_vnir_var.get()
        self._settings["process_swir"] = self._process_swir_var.get()
        _save_settings(self._settings)

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def _browse_input(self) -> None:
        initial = self._input_var.get()
        initial_dir = str(Path(initial).parent) if initial else self._settings.get("last_input_file", "")
        path = filedialog.askopenfilename(
            title="Selecteer invoerbestand",
            initialdir=initial_dir or None,
            filetypes=[
                ("ENVI headers", "*.hdr"),
                ("TIFF images", "*.tif *.tiff"),
                ("PNG images", "*.png"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._input_var.set(path)

    def _browse_output(self) -> None:
        initial = self._output_var.get() or self._settings.get("last_output_folder", "")
        path = filedialog.askdirectory(
            title="Selecteer uitvoermap",
            initialdir=initial or None,
        )
        if path:
            self._output_var.set(path)

    def _browse_input_folder(self) -> None:
        initial = self._input_var.get() or self._settings.get("last_input_file", "")
        path = filedialog.askdirectory(
            title="Selecteer invoer rootmap",
            initialdir=initial or None,
        )
        if path:
            self._input_var.set(path)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _log_msg(self, msg: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _start_convert(self) -> None:
        if self._running:
            return

        input_path = Path(self._input_var.get().strip())
        output_folder = Path(self._output_var.get().strip())

        if not input_path.name:
            messagebox.showwarning("Geen invoer", "Selecteer een invoerbestand of rootmap.")
            return
        if not input_path.exists():
            messagebox.showwarning("Pad niet gevonden", f"Kan pad niet vinden:\n{input_path}")
            return
        if not output_folder.name:
            messagebox.showwarning("Geen uitvoermap", "Selecteer een uitvoermap.")
            return

        output_folder.mkdir(parents=True, exist_ok=True)
        is_batch_mode = input_path.is_dir()
        output_path = output_folder / f"{input_path.stem}.las"

        self._collect_settings()

        rgb_vals = [v.get() for v in self._rgb_vars]
        config = ConversionConfig(
            input_path=input_path,
            output_path=output_path,
            use_envi_coordinates=self._envi_var.get(),
            x_meter_channel=int(self._envi_channel_vars[0].get()),
            x_fraction_channel=int(self._envi_channel_vars[1].get()),
            y_meter_channel=int(self._envi_channel_vars[2].get()),
            y_fraction_channel=int(self._envi_channel_vars[3].get()),
            z_meter_channel=int(self._envi_channel_vars[4].get()),
            z_fraction_channel=int(self._envi_channel_vars[5].get()),
            use_rgb_colors=self._rgb_var.get(),
            red_channel=int(rgb_vals[0]),
            green_channel=int(rgb_vals[1]),
            blue_channel=int(rgb_vals[2]),
            rgb_clip_low_percentile=float(rgb_vals[3]),
            rgb_clip_high_percentile=float(rgb_vals[4]),
            write_las=self._write_las_var.get(),
        )

        self._running = True
        self._is_batch_mode = is_batch_mode
        self._cancel_event = threading.Event()
        self._convert_btn.configure(state="disabled")
        self._cancel_btn.configure(state="normal" if is_batch_mode else "disabled")
        self._progress.start(10)

        if is_batch_mode:
            include_vnir = self._process_vnir_var.get()
            include_swir = self._process_swir_var.get()
            if not include_vnir and not include_swir:
                messagebox.showwarning("Geen type geselecteerd", "Selecteer minimaal VNIR of SWIR voor batchverwerking.")
                return

            self._log_msg(f"Scan bronmap: {input_path}")
            files = discover_envi_fused_hdr_files_filtered(
                input_path,
                include_vnir=include_vnir,
                include_swir=include_swir,
            )
            if not files:
                self._progress.stop()
                self._convert_btn.configure(state="normal")
                self._running = False
                messagebox.showwarning("Geen ENVI-bestanden", "Geen geselecteerde VNIR/SWIR .hdr bestanden gevonden in ENVI-fused submappen.")
                return
            self._log_msg(f"Scan klaar: {len(files)} bestanden gevonden.")
            self._log_msg(f"Batch-conversie gestart ({len(files)} bestanden) …")
            thread = threading.Thread(target=self._run_batch_conversion, args=(input_path, output_folder, config), daemon=True)
        else:
            if self._write_las_var.get():
                self._log_msg(f"Converteer {input_path.name} → {output_path} …")
            else:
                self._log_msg(f"Converteer {input_path.name} (zonder LAS schrijven) …")
            thread = threading.Thread(target=self._run_conversion, args=(config, output_path), daemon=True)
        thread.start()

    def _run_conversion(self, config: ConversionConfig, output_path: Path) -> None:
        try:
            convert_image_to_las(config)
            self.after(0, self._on_success, output_path, config.write_las)
        except Exception as exc:  # noqa: BLE001
            self.after(0, self._on_error, str(exc))

    def _run_batch_conversion(self, root_folder: Path, output_folder: Path, template_config: ConversionConfig) -> None:
        def _build_config(input_hdr: Path, output_las: Path) -> ConversionConfig:
            return ConversionConfig(
                input_path=input_hdr,
                output_path=output_las,
                channel=template_config.channel,
                x_scale=template_config.x_scale,
                y_scale=template_config.y_scale,
                z_scale=template_config.z_scale,
                x_offset=template_config.x_offset,
                y_offset=template_config.y_offset,
                z_offset=template_config.z_offset,
                invert_y=template_config.invert_y,
                band_index=template_config.band_index,
                x_band_index=template_config.x_band_index,
                y_band_index=template_config.y_band_index,
                z_band_index=template_config.z_band_index,
                use_envi_coordinates=template_config.use_envi_coordinates,
                x_meter_channel=template_config.x_meter_channel,
                x_fraction_channel=template_config.x_fraction_channel,
                y_meter_channel=template_config.y_meter_channel,
                y_fraction_channel=template_config.y_fraction_channel,
                z_meter_channel=template_config.z_meter_channel,
                z_fraction_channel=template_config.z_fraction_channel,
                use_rgb_colors=template_config.use_rgb_colors,
                red_channel=template_config.red_channel,
                green_channel=template_config.green_channel,
                blue_channel=template_config.blue_channel,
                rgb_clip_low_percentile=template_config.rgb_clip_low_percentile,
                rgb_clip_high_percentile=template_config.rgb_clip_high_percentile,
                write_las=template_config.write_las,
            )

        def _on_progress(index: int, total: int, input_hdr: Path) -> None:
            self.after(0, self._log_msg, f"Verwerk bestand {index}/{total}: {input_hdr.name}")

        try:
            result = convert_root_folder(
                root_folder,
                output_folder,
                _build_config,
                should_cancel=(lambda: self._cancel_event.is_set()) if self._cancel_event is not None else None,
                on_progress=_on_progress,
                include_vnir=self._process_vnir_var.get(),
                include_swir=self._process_swir_var.get(),
            )
            self.after(0, self._on_batch_done, result.converted, result.failed, result.cancelled)
        except Exception as exc:  # noqa: BLE001
            self.after(0, self._on_error, str(exc))

    def _on_success(self, output_path: Path, write_las: bool) -> None:
        self._progress.stop()
        self._convert_btn.configure(state="normal")
        self._cancel_btn.configure(state="disabled")
        self._running = False
        self._is_batch_mode = False
        self._cancel_event = None
        if write_las:
            self._log_msg(f"Klaar! LAS-bestand opgeslagen: {output_path}")
        else:
            self._log_msg("Klaar! LAS schrijven overgeslagen; rapporten/shapefiles zijn wel gemaakt.")

    def _on_error(self, message: str) -> None:
        self._progress.stop()
        self._convert_btn.configure(state="normal")
        self._cancel_btn.configure(state="disabled")
        self._running = False
        self._is_batch_mode = False
        self._cancel_event = None
        self._log_msg(f"FOUT: {message}")
        messagebox.showerror("Conversiefout", message)

    def _on_batch_done(self, converted: list[Path], failed: list[tuple[Path, str]], cancelled: bool) -> None:
        self._progress.stop()
        self._convert_btn.configure(state="normal")
        self._cancel_btn.configure(state="disabled")
        self._running = False
        self._is_batch_mode = False
        self._cancel_event = None
        self._log_msg(f"Batch klaar: {len(converted)} geconverteerd, {len(failed)} mislukt.")
        for output_path in converted[:10]:
            self._log_msg(f"OK: {output_path}")
        for input_path, message in failed[:10]:
            self._log_msg(f"FOUT: {input_path} -> {message}")
        if cancelled:
            self._log_msg("Batch geannuleerd door gebruiker.")
            messagebox.showinfo(
                "Batch geannuleerd",
                f"Verwerking gestopt. {len(converted)} bestanden geconverteerd, {len(failed)} fouten.",
            )
            return
        if failed:
            messagebox.showwarning(
                "Batch met fouten",
                f"{len(converted)} bestanden geconverteerd, {len(failed)} fouten. Zie log voor details.",
            )

    def _cancel_convert(self) -> None:
        if not self._running or not self._is_batch_mode:
            return
        if self._cancel_event is not None:
            self._cancel_event.set()
            self._cancel_btn.configure(state="disabled")
            self._log_msg("Annuleren aangevraagd: huidige bestand wordt afgemaakt, daarna stopt de batch.")


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
