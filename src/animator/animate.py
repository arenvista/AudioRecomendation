from __future__ import annotations
import re
from dataclasses import dataclass

import pandas as pd
from manim import *

@dataclass(frozen=True)
class CtClr:
    """Predefined RGB fraction colors for the PCA plot."""
    RED: tuple[float, float, float] = (210.0 / 255, 15.0 / 255, 57.0 / 255)
    ORANGE: tuple[float, float, float] = (254.0 / 255, 100.0 / 255, 11.0 / 255)
    YELLOW: tuple[float, float, float] = (223.0 / 255, 142.0 / 255, 29.0 / 255)
    GREEN: tuple[float, float, float] = (64.0 / 255, 160.0 / 255, 43.0 / 255)
    BLUE: tuple[float, float, float] = (32.0 / 255, 159.0 / 255, 181.0 / 255)
    PURPLE: tuple[float, float, float] = (114.0 / 255, 135.0 / 255, 253.0 / 255)
    WHITE: tuple[float, float, float] = (220.0 / 255, 224.0 / 255, 232.0 / 255)


class PCABaseScene(ThreeDScene):
    """
    Base class that handles data loading, 3D environment setup, 
    dot generation, and HUD/Legend management.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.legend_data: dict[str, str] = {}
        self.legend_mobject: Group | None = None
        self.dot_map: dict[str, Dot3D] = {}
        self.dots = Group()
        self.axes: ThreeDAxes | None = None
        self.pca_df: pd.DataFrame = pd.DataFrame()
        self.curr_time = 0

    def wait_update_time(self, min_in_vid:int, sec_in_vid: int ):
        block = ((min_in_vid*60) + sec_in_vid) - self.curr_time
        self.curr_time += block
        self.wait(block)

    def measure_distance(self, filename1: str, filename2: str, line_color, txt=None):
        """Draws a line between two dots and displays their exact PCA distance."""
        dot1 = self.dot_map[filename1]
        dot2 = self.dot_map[filename2]

        # 1. Get visual coordinates for drawing the line
        pos1 = dot1.get_center()
        pos2 = dot2.get_center()

        # 2. Get actual data coordinates for the math
        row1 = self.pca_df[self.pca_df['Filename'] == filename1].iloc[0]
        row2 = self.pca_df[self.pca_df['Filename'] == filename2].iloc[0]
        
        data_coords1 = np.array([row1['PC1'], row1['PC2'], row1['PC3']])
        data_coords2 = np.array([row2['PC1'], row2['PC2'], row2['PC3']])
        
        # Calculate Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
        actual_distance = np.linalg.norm(data_coords2 - data_coords1)

        # 3. Draw a dashed line between the dots
        connection = Line3D(start=pos1, end=pos2, color=line_color, thickness=0.02)

        # 4. Create the text label at the midpoint
        midpoint = (pos1 + pos2) / 2
        dist_text = f"{actual_distance:.2f}"
        if txt is None:
            self.legend_data[f"Distance({filename1},{filename2})=[{dist_text}]"] = line_color
        else:
            self.legend_data[f"Distance({txt})=[{dist_text}]"] = line_color
        self._redraw_legend()

        # 5. Animate
        # self.play(Create(connection), run_time=1.5)
        self.add(connection) # Explicitly ensures it stays in the scene's object list

    def setup_base_environment(self, file_path: str, animate: bool = True):
        """Loads data, draws axes, generates dots, and adds labels."""
        
        self.pca_df = pd.read_csv(file_path)

        # Find the absolute min and max across all three dimensions
        min_val = self.pca_df[['PC1', 'PC2', 'PC3']].min().min()
        max_val = self.pca_df[['PC1', 'PC2', 'PC3']].max().max()
        
        # Add 10% padding so dots don't sit exactly on the edge
        padding = (max_val - min_val) * 0.1
        axis_min = min_val - padding
        axis_max = max_val + padding
        
        # Determine a sensible step size (roughly 5-10 tick marks)
        step = (axis_max - axis_min) / 5

        # Setup Axes using dynamic ranges
        self.axes = ThreeDAxes(
            x_range=[axis_min, axis_max, step], 
            y_range=[axis_min, axis_max, step], 
            z_range=[axis_min, axis_max, step],
            x_length=8, y_length=8, z_length=4,
            # Clean up the look by adjusting tip width or tick size if desired
            axis_config={"tick_size": 0.05, "tip_width": 0.15, "tip_height": 0.15}
        )
        
        # --- NEW: Add Axis Labels ---
        # Generate the text labels for the ends of the axes
        self.axis_labels = self.axes.get_axis_labels(
            x_label=Text("PC1", font_size=24), 
            y_label=Text("PC2", font_size=24), 
            z_label=Text("PC3", font_size=24)
        )
        
        # Lighting & Camera Fix
        self.set_camera_orientation(phi=85 * DEGREES, theta=30 * DEGREES)
        self.camera.focal_distance = 20
        
        self.wait_update_time(0,33)
        # self.wait(33)
        # Generate dots based on standard color
        self._create_dots(color_standard=CtClr.BLUE)

        if animate:
            # Animate the axes and labels appearing together
            self.play(
                Create(self.axes),
                FadeIn(self.axis_labels, shift=OUT * 0.5)
            )
            self.play(LaggedStart(*[FadeIn(dot) for dot in self.dots], lag_ratio=0.01))
        else:
            self.add(self.axes, self.axis_labels, self.dots)

    def _create_dots(self, color_standard: tuple[float, float, float] | None = None):
        """Internal method to parse dataframe and populate the 3D dots."""
        # Pre-compile regex for significant performance boost inside the loop
        color_pattern = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")

        for _, row in self.pca_df.iterrows():
            x, y, z = row['PC1'], row['PC2'], row['PC3']
            
            if color_standard is None: 
                color_str = str(row['Point_Color'])
                rgba_vals = [float(val) for val in color_pattern.findall(color_str)]
                dot_color = rgb_to_color(rgba_vals[:3])
            else: 
                dot_color = rgb_to_color(color_standard)
            
            dot = Dot3D(
                point=self.axes.c2p(x, y, z),
                radius=0.10,
                color=dot_color,
                gloss=0.8,      
                shadow=0.2,     
            )
            self.dots.add(dot)
            self.dot_map[row["Filename"]] = dot

    def highlight_dot(self, filename: str, color: tuple[float, float, float], label: str | None = None):
        """Updates dot color by filename and manages the 2D legend."""
        dot = self.dot_map[filename]
        print(dot, filename)

        dot_color = rgb_to_color(color)
        dot.set_color(dot_color)

        # Skip legend update if no label is provided
        if label is None:
            return

        # Add to legend tracker if unique
        if label not in self.legend_data:
            self.legend_data[label] = dot_color
            self._redraw_legend()

    def _redraw_legend(self):
        """Refreshes the on-screen legend."""
        if self.legend_mobject is not None:
            self.remove(self.legend_mobject)

        entries = Group()
        for label, col in self.legend_data.items():
            key_dot = Dot(color=col, radius=0.03)
            key_text = Text(str(label), color=WHITE).scale(0.08)
            item = Group(key_dot, key_text)
            key_text.next_to(key_dot, RIGHT, buff=0.15)
            entries.add(item)

        entries.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        entries.to_corner(DR, buff=0.5)

        bg = SurroundingRectangle(entries, color=BLACK, fill_opacity=0.5, buff=0.15)
        self.legend_mobject = Group(bg, entries)

        self.add_fixed_in_frame_mobjects(self.legend_mobject)


# --- Example Scenes ---

class PCASpectrogramDrawDots(PCABaseScene):
    def construct(self):
        file_path = "./output/pca/2026-04-09_09-02-54_pca.csv.augmented.csv.colored.csv"
        
        self.begin_ambient_camera_rotation(rate=0.2)
        self.setup_base_environment(file_path=file_path, animate=True)
        self.wait_update_time(0,48)
        # Use the new encapsulated highlight method

        self.highlight_dot("505.mp3", CtClr.RED, "505")
        self.wait_update_time(1,7) # EoS

        self.highlight_dot("September.mp3", CtClr.ORANGE, "Sept")
        self.wait_update_time(1,24) # EoS

        self.measure_distance("505.mp3", "September.mp3", RED, "505-Sept") 
        self.wait_update_time(1,37) # EoEx

        # Show examples that are close 
        # Both By Askew Ed; Piano
        self.highlight_dot("000193.mp3", CtClr.YELLOW, "Here With You")
        self.highlight_dot("000207.mp3", CtClr.GREEN, "Piano")
        self.wait_update_time(2,13) # EoS

        self.measure_distance("000193.mp3", "000207.mp3", GOLD_A, "Piano-Piano") 
        self.wait_update_time(2,40) # EoEx
        

        self.highlight_dot("001040.mp3", CtClr.WHITE, "Dracula Mountain") 
        self.wait_update_time(2,46) # EoEx
        self.highlight_dot("002012.mp3", CtClr.PURPLE, "The White Mice") 
        self.wait_update_time(2,48) # EoEx
        self.measure_distance("001040.mp3", "002012.mp3", PURPLE, "Rock-Rock")

        self.wait_update_time(3,16) # EoEx
        self.measure_distance("001040.mp3", "000207.mp3", PURPLE, "Piano-Rock")

        self.wait_update_time(4,15) # EoEx
        self.stop_ambient_camera_rotation()


class PCASpectrogramPointsOfInterest(PCABaseScene):
    def construct(self):
        file_path = "./output/pca/2026-04-09_09-02-54_pca.csv.augmented.csv.colored.csv"
        
        self.setup_base_environment(file_path=file_path, animate=False)
        
        last_row = self.pca_df.iloc[-1]
        x, y, z = last_row['PC1'], last_row['PC2'], last_row['PC3']
        target_point = self.axes.c2p(x, y, z)

        self.move_camera(focal_point=target_point, zoom=5, phi=80 * DEGREES, run_time=2.5)

        point_name = str(last_row["Filename"])
        label = Text(point_name, font_size=24).scale(0.5)
        label.rotate(PI / 2, axis=RIGHT)
        label.rotate(30 * DEGREES, axis=OUT)
        label.move_to(target_point + OUT * 0.4) 
        
        self.play(FadeIn(label))
        self.wait(3)

class AutoencoderVisual(Scene):
    def construct(self):
        # 1. Define the architecture of the Autoencoder
        # Input (7) -> Hidden (5) -> Latent Bottleneck (2) -> Hidden (5) -> Output (7)
        layer_sizes = [7, 5, 2, 5, 7]
        layers = VGroup()
        
        # Create the nodes for each layer
        for i, size in enumerate(layer_sizes):
            layer = VGroup(*[Circle(radius=0.15, color=WHITE) for _ in range(size)])
            layer.arrange(DOWN, buff=0.3)
            layers.add(layer)
            
        layers.arrange(RIGHT, buff=1.5)
        
        # Color coding the sections
        encoder_color = BLUE
        latent_color = YELLOW
        decoder_color = GREEN
        
        layers[0].set_color(encoder_color)  # Input layer
        layers[1].set_color(encoder_color)  # Encoder hidden layer
        layers[2].set_color(latent_color)   # Latent Space / Bottleneck
        layers[3].set_color(decoder_color)  # Decoder hidden layer
        layers[4].set_color(decoder_color)  # Output layer

        # 2. Create the edges (connections) between nodes
        edges = VGroup()
        for i in range(len(layers) - 1):
            layer_edges = VGroup()
            for node1 in layers[i]:
                for node2 in layers[i+1]:
                    edge = Line(node1.get_center(), node2.get_center(), stroke_width=1, stroke_opacity=0.3)
                    layer_edges.add(edge)
            edges.add(layer_edges)

        # 3. Add Labels
        title = Text("Autoencoder Architecture", font_size=40).to_edge(UP)
        
        enc_label = Text("Encoder", font_size=24, color=encoder_color).next_to(layers[0:2], UP, buff=0.5)
        lat_label = Text("Latent Space", font_size=24, color=latent_color).next_to(layers[2], UP, buff=0.5)
        dec_label = Text("Decoder", font_size=24, color=decoder_color).next_to(layers[3:5], UP, buff=0.5)

        # --- ANIMATION SEQUENCE ---

        # Animate Title
        self.play(Write(title))
        self.wait(0.5)

        # Animate Encoder
        self.play(Write(enc_label))
        self.play(Create(layers[0]), Create(layers[1]))
        self.play(Create(edges[0]))
        self.wait(0.5)

        # Animate Latent Space (Bottleneck)
        self.play(Write(lat_label))
        self.play(Create(layers[2]))
        self.play(Create(edges[1]))
        self.play(Indicate(layers[2], scale_factor=1.5, color=YELLOW))
        self.wait(0.5)

        # Animate Decoder
        self.play(Write(dec_label))
        self.play(Create(layers[3]), Create(layers[4]))
        self.play(Create(edges[2]), Create(edges[3]))
        self.wait(1)

        # 4. Animate Data Flow (Forward Pass)
        # Create glowing dots representing data flowing through the network
        self.play(title.animate.become(Text("Forward Pass: Compression & Reconstruction", font_size=40).to_edge(UP)))
        
        flow_animations = []
        for i in range(len(layers) - 1):
            layer_flow = []
            for node2 in layers[i+1]:
                # Pick a random node from the previous layer to originate from for visual clarity
                # Alternatively, you can pulse the edges, but passing dots looks cleaner
                for node1 in layers[i]:
                    dot = Dot(color=YELLOW, radius=0.08)
                    dot.move_to(node1.get_center())
                    layer_flow.append(MoveAlongPath(dot, Line(node1.get_center(), node2.get_center())))
            
            # Play the flow between the two layers simultaneously
            self.play(*layer_flow, run_time=1)
            
            # Highlight the nodes being activated
            self.play(layers[i+1].animate.set_fill(layers[i+1][0].get_color(), opacity=0.8), run_time=0.3)
            self.play(layers[i+1].animate.set_fill(opacity=0), run_time=0.3)

        self.wait(1)

        # Highlight Input vs Output comparison
        compare_text = Text("Input ≈ Output", font_size=36, color=WHITE).next_to(layers, DOWN, buff=0.5)
        self.play(Write(compare_text))
        
        # Flash the input and output layers to show their relationship
        self.play(
            Indicate(layers[0], color=encoder_color, scale_factor=1.2),
            Indicate(layers[4], color=decoder_color, scale_factor=1.2),
            run_time=2
        )

        self.wait(2)
        
        # Clean up
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(1)

from manim import *
import numpy as np

class MP3ToSpectrogramConceptual(Scene):
    def construct(self):
        # 1. Setup simulated audio signal (Waveform)
        duration = 10
        fs = 100  # Sampling rate for visualization
        t = np.linspace(0, duration, duration * fs, endpoint=False)
        
        # A complex signal changing over time: low frequency base, higher frequency chirps
        signal = (
            np.sin(2 * np.pi * 0.8 * t) +              
            0.5 * np.sin(2 * np.pi * 3.0 * t * (1 + 0.1 * t)) + 
            0.3 * np.cos(2 * np.pi * 7.0 * t - 0.5 * t * t)  
        )
        
        # Manim Axes for the Waveform (Top Left)
        waveform_axes = Axes(
            x_range=[0, duration, 2],
            y_range=[-2.5, 2.5, 1],
            axis_config={"include_numbers": True},
        ).scale(0.5).to_corner(UL, buff=0.0)

        waveform_labels = waveform_axes.get_axis_labels(
            x_label=Text("Time (s)", font_size=14),
            y_label=Text("Amplitude", font_size=14)
        )

        waveform_graph = waveform_axes.plot_line_graph(
            x_values=t,
            y_values=signal,
            line_color=BLUE,
            add_vertex_dots=False,
            stroke_width=1.5
        )

        waveform_title = Text("Decoded MP3 Data", font_size=20, color=BLUE).next_to(waveform_axes, UP, buff=0.15)

        # 2. Add Zoomed Axes Setup (Top Right, Side-by-Side)
        num_time_bins = 10
        window_time_width = duration / num_time_bins

        zoom_axes = Axes(
            x_range=[0, window_time_width, 0.5],
            y_range=[-2.5, 2.5, 1],
            axis_config={"include_numbers": True},
        ).scale(0.5).to_corner(UR, buff=0.0)

        zoom_labels = zoom_axes.get_axis_labels(
            x_label=Text("Time (s)", font_size=14),
            y_label=Text("Amplitude", font_size=14)
        )
        
        zoom_title = Text("Zoomed Window", font_size=20, color=RED).next_to(zoom_axes, UP, buff=0.15)

# 3. Add Spectrogram Grid Setup (Moved Down and Left)
        num_freq_bins = 12
        window_size_viz = (duration * fs) // num_time_bins
        
        spec_axes = Axes(
            x_range=[0, duration, duration // num_time_bins],
            y_range=[0, num_freq_bins, num_freq_bins // 6],
            x_axis_config={"include_numbers": True},
            y_axis_config={"include_numbers": True},
        ).scale(0.50).to_corner(DL, buff=0.2) # <-- Changed to Down-Left corner

        spec_labels = spec_axes.get_axis_labels(
            x_label=Text("Time (s)", font_size=6),
            y_label=Text("Frequency (Hz)", font_size=6)
        )

        spec_title = Text("Constructing Spectrogram", font_size=12, color=YELLOW).next_to(spec_axes, UP, buff=0.1)

        # Initialize an empty grid of rectangles
        spec_grid = VGroup()
        for i in range(num_time_bins):
            for j in range(num_freq_bins):
                x_pos = spec_axes.c2p(i * window_time_width, j + 0.5)
                rect = Rectangle(
                    width=spec_axes.x_axis.get_unit_size() * window_time_width,
                    height=spec_axes.y_axis.get_unit_size(),
                    stroke_color=WHITE,
                    stroke_width=0.5,
                    fill_opacity=0,
                )
                rect.move_to(x_pos + RIGHT * (spec_axes.x_axis.get_unit_size() * window_time_width / 2))
                spec_grid.add(rect)

        # Create a rectangle scaled perfectly to the axes for the moving window
        moving_window = Rectangle(
            width=waveform_axes.x_axis.get_unit_size() * window_time_width,
            height=waveform_axes.y_axis.get_length(),
            color=RED,
            stroke_width=2
        ).move_to(waveform_axes.c2p(window_time_width / 2, 0))

        # Labels and middle visualization for the transformation process
        stft_text = Text("Short-Time Fourier Transform", font_size=12, color=RED).next_to(waveform_axes, DOWN, buff=0.3)
        freq_decomposition_title = Text("Frequency Components", font_size=12).next_to(zoom_axes, DOWN, buff=3.5)

        bars_decomposition = VGroup(*[
            Rectangle(width=0.12, height=0.1, fill_color=YELLOW, fill_opacity=0.8, stroke_opacity=0)
            for _ in range(num_freq_bins)
        ]).arrange(RIGHT, buff=0.08)
        bars_decomposition.next_to(freq_decomposition_title, DOWN, buff=-0.6)

        # Initial Animation Sequence
        self.play(Write(waveform_title), Create(waveform_axes), Write(waveform_labels))
        self.play(Create(waveform_graph), run_time=2)
        self.wait(0.5)

        self.play(Write(stft_text), Create(moving_window))
        self.play(FadeIn(zoom_axes), FadeIn(zoom_labels), FadeIn(zoom_title))
        self.play(Create(spec_axes), Write(spec_labels), Write(spec_title))
        self.play(Create(spec_grid), run_time=1)
        self.play(FadeIn(freq_decomposition_title), FadeIn(bars_decomposition))
        self.wait(1)

        # 4. Animated Transformation Cycle
        def simulate_fft_results(t_slice):
            t_center = np.mean(t_slice)
            fft_profile = np.zeros(num_freq_bins)
            fft_profile[0] = 0.9  
            
            ramping_bin = int(min(11, 2 + 1.2 * t_center))
            fft_profile[ramping_bin] = 0.7 * (t_center/10) 
            fft_profile[max(0, ramping_bin-1)] = 0.3 

            dropping_bin = int(max(0, 8 - 1.0 * t_center))
            fft_profile[dropping_bin] = 0.5 * (1 - t_center/12) 
            
            fft_profile += np.random.rand(num_freq_bins) * 0.1
            return np.clip(fft_profile, 0, 1)

        intensity_cmap = lambda x: interpolate_color(PURPLE_E, YELLOW, x)

        num_slow_windows = 3
        slice_graph_target = None  

        for i in range(num_slow_windows):
            idx_start = i * window_size_viz
            idx_end = (i + 1) * window_size_viz
            t_slice = t[idx_start:idx_end]
            sig_slice = signal[idx_start:idx_end]
            
            # 1. Slide window
            t_center = np.mean(t_slice)
            self.play(moving_window.animate.move_to(waveform_axes.c2p(t_center, 0)), run_time=1)

            # 2. Extract slice to the side-by-side zoom axes
            slice_graph = waveform_axes.plot_line_graph(
                x_values=t_slice, y_values=sig_slice, add_vertex_dots=False, line_color=RED, stroke_width=2
            )
            # Plot directly on zoom_axes, shifting time to 0
            slice_graph_target = zoom_axes.plot_line_graph(
                x_values=t_slice - t_slice[0], y_values=sig_slice, add_vertex_dots=False, line_color=RED, stroke_width=2
            )
            
            self.play(ReplacementTransform(slice_graph, slice_graph_target), run_time=1)
            
            # 3. Simulate FFT -> Update Bars
            fft_data = simulate_fft_results(t_slice)
            bar_updates = []
            for j in range(num_freq_bins):
                bar_height = max(0.05, 1.8 * fft_data[j])
                bar_updates.append(
                    bars_decomposition[j].animate.stretch_to_fit_height(bar_height).set_color(intensity_cmap(fft_data[j])).align_to(bars_decomposition, DOWN)
                )
            
            self.play(*bar_updates, run_time=1)

            # 4. Transfer FFT Columns -> Stack Spectrogram
            final_col = VGroup()
            for j in range(num_freq_bins):
                grid_idx = i * num_freq_bins + j
                cell_rect = spec_grid[grid_idx]
                cell_rect.set_fill(color=intensity_cmap(fft_data[j]), opacity=0.9)
                cell_rect.set_stroke(width=0)
                final_col.add(cell_rect.copy())

            self.play(Indicate(final_col, color=YELLOW, scale_factor=1.1), FadeOut(slice_graph_target), run_time=1)

        # 5. Faster Fill for the rest
        rest_of_stft = Text("FFT of Remaining...", font_size=16).move_to(freq_decomposition_title)
        self.play(Transform(freq_decomposition_title, rest_of_stft))
        
        fill_remaining = []
        for i in range(num_slow_windows, num_time_bins):
            idx_start = i * window_size_viz
            idx_end = (i + 1) * window_size_viz
            t_slice = t[idx_start:idx_end]
            fft_data = simulate_fft_results(t_slice)
            waveform_pos_target = waveform_axes.c2p(np.mean(t_slice), 0)
            
            slice_group = VGroup()
            for j in range(num_freq_bins):
                grid_idx = i * num_freq_bins + j
                cell_rect = spec_grid[grid_idx]
                cell_rect.set_fill(color=intensity_cmap(fft_data[j]), opacity=0.9)
                cell_rect.set_stroke(width=0)
                slice_group.add(cell_rect)
                
            fill_remaining.append(AnimationGroup(
                moving_window.animate.move_to(waveform_pos_target),
                FadeIn(slice_group),
                lag_ratio=0
            ))

        # Use Succession to play them sequentially for a smooth sweep
        self.play(Succession(*fill_remaining), run_time=3)
        self.wait(1)

        # 6. Final Scene comparison
        result_text = Text("Result: Time-Frequency Map", font_size=24, color=YELLOW).next_to(spec_axes, UP, buff=0.15)
        
        self.play(
            FadeOut(stft_text), 
            FadeOut(moving_window), 
            FadeOut(freq_decomposition_title), 
            FadeOut(bars_decomposition),
            FadeOut(zoom_axes),
            FadeOut(zoom_title),
            FadeOut(zoom_labels)
        )
        
        self.play(
            ReplacementTransform(spec_title, result_text),
            Indicate(spec_grid, scale_factor=1.05, color=YELLOW)
        )
        self.wait(2)
