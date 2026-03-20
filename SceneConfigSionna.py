import os
os.environ.pop("TF_USE_LEGACY_KERAS", None)   # Or set os.environ["TF_USE_LEGACY_KERAS"] = "0"
import gc

if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
# from antenna_vsat import v60g_tx_pattern_x_axis
from satellite_projection import satellite_projection
from sionnautils.miutils import CoverageMapPlanner 
from sionna.rt import PlanarArray, Transmitter, Receiver, PathSolver
# from sionna.channel import cir_to_time_channel
# from sionna.rt.antenna import visualize
tf.keras.backend.clear_session()
from scipy.special import jv
import mitsuba as mi
from sionna.rt.antenna_pattern import register_antenna_pattern




# Define constants (adjust these as needed)
dish_diameter = 0.30        # meters
tx_gain_dB = 38.1           # dBi
c = 3e8                     # speed of light (m/s)
tx_frequency_mid = np.mean([13.75, 14.5]) * 1e9  # mid frequency in Hz
def v_vsat_pattern(theta: mi.Float, phi: mi.Float) -> mi.Complex2f:
    """
    Custom vertically-polarized VSAT antenna pattern.
    
    This function uses the (2*J₁(u)/u)² model, where
    u = π*dish_diameter*sin(angle_from_boresight)/(λ)
    and applies gain normalization and back-lobe suppression.
    """
    # Convert theta and phi to numpy arrays (if not already)
    theta_np = theta.numpy() if hasattr(theta, "numpy") else np.array(theta)
    phi_np = phi.numpy() if hasattr(phi, "numpy") else np.array(phi)
    
    # Compute the angle from the boresight.
    # Here, boresight is assumed to point along the positive x-axis.
    ray_x = np.sin(theta_np) * np.cos(phi_np)
    angle_from_x = np.arccos(np.clip(ray_x, -1.0, 1.0))
    
    # Compute u parameter using the mid-frequency wavelength λ = c/tx_frequency_mid
    u = np.pi * dish_diameter * np.sin(angle_from_x) / (c / tx_frequency_mid)
    epsilon = 1e-10
    u_safe = np.where(np.abs(u) < epsilon, epsilon, u)
    
    # Compute the aperture pattern using the Bessel function J₁
    j1_u = jv(1, u_safe)
    pattern = (2.0 * j1_u / u_safe)**2
    # Handle boresight (when u is zero)
    pattern = np.where(np.abs(u) < epsilon, 1.0, pattern)
    
    # Apply gain normalization (convert from dBi to linear scale)
    max_gain_linear = 10**(tx_gain_dB/10)
    normalized_pattern = pattern * max_gain_linear
    
    # Suppress the back-lobe (for angles beyond 90° from boresight)
    back_lobe_mask = angle_from_x > (np.pi/2)
    normalized_pattern = np.where(back_lobe_mask, normalized_pattern * 0.001, normalized_pattern)
    
    # Convert from power to field amplitude (E = √power)
    field_amplitude = np.sqrt(normalized_pattern)
    
    # Convert the result into the expected complex type (imaginary part is zero)
    field_amplitude_mi = mi.Float(field_amplitude)
    return mi.Complex2f(field_amplitude_mi, mi.Float(0))

def create_vsat_factory(name: str):
    def f(*, polarization, polarization_model="tr38901_2"):
        from sionna.rt.antenna_pattern import PolarizedAntennaPattern
        return PolarizedAntennaPattern(
            v_pattern=globals()["v_" + name + "_pattern"],
            polarization=polarization,
            polarization_model=polarization_model
        )
    return f

# Ensure your custom pattern function is available in globals
globals()["v_vsat_pattern"] = v_vsat_pattern

# Register the antenna pattern with the name "vsat"
register_antenna_pattern("vsat", create_vsat_factory("vsat"))

class SceneConfigSionna:
    def __init__(self, scene):
        """
        scene : A sionna.rt.Scene object with loaded geometry (XML or otherwise).
        """
        self.scene = scene
        
        self.fc = 10e9

        # Some default parameters you can modify:
        self.grid_size = 1.0
        self.nbs = None               # Number of gNB/base stations
        self.nsect = None              # Usually 3, for sector-based coverage
        # self.BS_height_above_roof = 35  # For base station on building
        # self.BS_height_above_ground = 45
        self.BS_height_above_roof = 45  # For base station on building
        self.BS_height_above_ground = 55
        self.tn_height_above_roof = 1.2  # For base station on building
        self.tn_height_above_ground = 1.8
        self.ntn_height_above_roof = 1.2  # For base station on building
        self.ntn_height_above_ground = 1.88
        self.sat_distance = 500e3   # Satellite distance from region (m)

        # Coverage map placeholders
        self.cm = None
        self.L_NS = None
        self.W_WE = None
        self.bbox = None
        self.extent = None
        self.point_type = None
        # Position arrays
        self.tx_pos = None
        self.rx_ntn_pos = None
        self.tn_pos = None
        self.ntn_look_pos = None

        # Path results
        self.a_bs_to_tn = None
        self.tau_bs_to_tn = None
        self.a_tn_to_bs = None
        self.tau_tn_to_bs = None
        self.a_ntn_to_bs = None
        self.tau_ntn_to_bs = None
        self.a_ntn_to_tn = None
        self.tau_ntn_to_tn = None
        self.ntn_rx = None
        
        self.tn_bs_index = None
        self.tn_sector_index = None
        self.tx_bs_index = None
        self.tx_sector_index = None
        self.tx_orientation_rad = None
        self.tx_name_list = None
        # Default BS sector orientation controls (can be overwritten from notebook)
        self.tx_sector_yaw_offset_rad = 0.0
        self.tx_sector_pitch_rad = -0.174533  # 10 deg down-tilt
        self.tx_sector_roll_rad = 0.0

    def build_coverage_map(self, grid_size=None, show_xy=False, plot=False):
        """
        Build coverage map, compute bbox/extent/point_type.
        Optional: print x/y ranges and plot building/outdoor map.
        """
        if grid_size is not None:
            self.grid_size = grid_size

        self.cm = CoverageMapPlanner(self.scene._scene, grid_size=self.grid_size)
        self.cm.set_grid()
        self.cm.compute_grid_attributes()
        self.hm = None

        x_min, x_max = self.cm.x[0], self.cm.x[-1]
        y_min, y_max = self.cm.y[0], self.cm.y[-1]

        self.W_WE = x_max - x_min   # width (East-West)
        self.L_NS = y_max - y_min   # length (North-South)
        self.bbox = [-self.W_WE/2, self.W_WE/2, -self.L_NS/2, self.L_NS/2]
        # self.extent = [self.cm.x[0], self.cm.x[-1], self.cm.y[0], self.cm.y[-1]]
        self.extent = [x_min, x_max, y_min, y_max]

        building = (self.cm.bldg_grid)
        self.point_type = np.where(building, 2, 1).astype(np.int8)
        self.point_type = np.flipud(self.point_type)
        # outdoor = (self.cm.bldg_grid == False)
        # self.point_type = outdoor + 2 * building
        # self.point_type = self.point_type.astype(int)
        # self.point_type = np.flipud(self.point_type)

        if show_xy:
            print(f"x range: [{x_min:.3f}, {x_max:.3f}]")
            print(f"y range: [{y_min:.3f}, {y_max:.3f}]")

        if plot:
            self._plot_grid()

    def _plot_grid(self, show_bs=False, show_tn=False, show_ntn=False):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        colors = ['lightgray', 'brown']
        cmap = ListedColormap(colors)
        plt.figure()
        plt.imshow(self.point_type, cmap=cmap, interpolation='nearest', extent=self.extent)
        if show_bs and self.tx_pos is not None:
            bs_marker = (3, 0, -30)  # triangle rotated 30 degrees clockwise
            plt.scatter(self.tx_pos[:, 0], self.tx_pos[:, 1], c="red", s=60, marker=bs_marker, label="BS")
        if show_tn and self.tn_pos is not None:
            plt.scatter(self.tn_pos[:, 0], self.tn_pos[:, 1], c="green", s=20, marker="o", label="TN")
        if show_ntn and self.rx_ntn_pos is not None:
            plt.scatter(self.rx_ntn_pos[:, 0], self.rx_ntn_pos[:, 1], c="blue", s=20, marker="x", label="NTN")
        has_overlays = (
            (show_bs and self.tx_pos is not None)
            or (show_tn and self.tn_pos is not None)
            or (show_ntn and self.rx_ntn_pos is not None)
        )
        if has_overlays:
            plt.legend(loc="upper right")
        plt.xlabel("x")
        plt.ylabel("y")
        if has_overlays:
            plt.title("Spatial Distribution of TN/NTN UEs and BSs")
        else:
            plt.title("Building/Outdoor Map")
        plt.show()

    def _clear_radio_nodes(self):
        for rx_name in list(self.scene.receivers):
            self.scene.remove(rx_name)
        for tx_name in list(self.scene.transmitters):
            self.scene.remove(tx_name)

    def _make_planar_array(self, num_rows, num_cols, pattern, polarization="V"):
        return PlanarArray(
            num_rows=int(num_rows),
            num_cols=int(num_cols),
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            polarization=polarization,
            pattern=pattern,
        )

    def _best_effort_rt_memory_cleanup(self):
        gc.collect()
        try:
            import drjit as dr

            if hasattr(dr, "sync_thread"):
                dr.sync_thread()
            if hasattr(dr, "flush_malloc_cache"):
                dr.flush_malloc_cache()
        except Exception:
            pass

    def _solve_cir_numpy(self, p_solver, *, max_depth):
        paths = p_solver(
            scene=self.scene,
            max_depth=max_depth,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            synthetic_array=True,
        )
        try:
            return paths.cir(normalize_delays=False, out_type="numpy")
        finally:
            del paths
            self._best_effort_rt_memory_cleanup()

    def _solve_cir_for_tx_batches(
        self,
        *,
        p_solver,
        max_depth,
        tx_indices,
        tx_batch_size,
        add_tx_batch_fn,
        add_rx_fn,
    ):
        tx_indices = np.asarray(tx_indices, dtype=int)
        if tx_indices.size == 0:
            return None, None

        if tx_batch_size is None:
            tx_batch_size = int(tx_indices.size)
        else:
            tx_batch_size = max(1, int(tx_batch_size))

        if tx_batch_size >= int(tx_indices.size):
            self._clear_radio_nodes()
            add_tx_batch_fn(tx_indices)
            add_rx_fn()
            try:
                return self._solve_cir_numpy(p_solver, max_depth=max_depth)
            finally:
                self._clear_radio_nodes()
                self._best_effort_rt_memory_cleanup()

        a_chunks = []
        for start in range(0, int(tx_indices.size), tx_batch_size):
            batch = tx_indices[start : start + tx_batch_size]
            self._clear_radio_nodes()
            add_tx_batch_fn(batch)
            add_rx_fn()
            try:
                a_chunk, _tau_chunk = self._solve_cir_numpy(p_solver, max_depth=max_depth)
            finally:
                self._clear_radio_nodes()
                self._best_effort_rt_memory_cleanup()
            a_chunks.append(np.asarray(a_chunk))

        if not a_chunks:
            return None, None
        return np.concatenate(a_chunks, axis=2), None

    def _make_receiver(self, *, name, position, color=None, orientation=None):
        kwargs = {"name": name, "position": position}
        if color is not None:
            kwargs["color"] = color
        if orientation is not None:
            try:
                return Receiver(orientation=orientation, **kwargs)
            except TypeError:
                rx = Receiver(**kwargs)
                try:
                    rx.orientation = orientation
                except Exception:
                    pass
                return rx
        return Receiver(**kwargs)

    def _prepare_bs_sector_state(self):
        if self.tx_pos is None or self.tx_pos.shape[0] == 0:
            raise ValueError("tx_pos is empty; run compute_positions first.")
        if self.nsect is None:
            raise ValueError("nsect is not set; configure sector metadata first.")

        sector_yaw = np.mod(
            float(self.tx_sector_yaw_offset_rad) + 2.0 * np.pi * np.arange(self.nsect) / self.nsect,
            2.0 * np.pi,
        )

        tx_name_list = []
        tx_bs_index = []
        tx_sector_index = []
        tx_orientation = []
        for bs_idx in range(self.tx_pos.shape[0]):
            for sec_idx in range(self.nsect):
                tx_name_list.append(f"tx-{bs_idx}-{sec_idx}")
                tx_bs_index.append(int(bs_idx))
                tx_sector_index.append(int(sec_idx))
                tx_orientation.append(
                    [
                        float(sector_yaw[sec_idx]),
                        float(self.tx_sector_pitch_rad),
                        float(self.tx_sector_roll_rad),
                    ]
                )

        self.tx_name_list = tx_name_list
        self.tx_bs_index = np.asarray(tx_bs_index, dtype=int)
        self.tx_sector_index = np.asarray(tx_sector_index, dtype=int)
        self.tx_orientation_rad = np.asarray(tx_orientation, dtype=float)

        if self.tn_pos is None or self.tn_pos.shape[0] == 0:
            self.tn_bs_index = np.empty((0,), dtype=int)
            self.tn_sector_index = np.empty((0,), dtype=int)
            return

        diffs = self.tn_pos[:, None, :2] - self.tx_pos[None, :, :2]
        d2 = np.sum(diffs**2, axis=2)
        self.tn_bs_index = np.argmin(d2, axis=1)

        bs_xy = self.tx_pos[:, :2]
        tn_xy = self.tn_pos[:, :2]
        rel = tn_xy - bs_xy[self.tn_bs_index]
        ang = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2 * np.pi)
        dtheta = np.abs(ang[:, None] - sector_yaw[None, :])
        dtheta = np.minimum(dtheta, 2 * np.pi - dtheta)
        self.tn_sector_index = np.argmin(dtheta, axis=1)

    def configure_cir_scene(
        self,
        *,
        nsect,
        fc=None,
        bandwidth=100e6,
        sector_yaw_offset_rad=None,
        sector_pitch_rad=None,
        sector_roll_rad=None,
    ):
        """Configure the scene-wide metadata used by subsequent CIR solves."""
        self.nsect = int(nsect)
        if fc is not None:
            self.fc = float(fc)
        if sector_yaw_offset_rad is not None:
            self.tx_sector_yaw_offset_rad = float(sector_yaw_offset_rad)
        if sector_pitch_rad is not None:
            self.tx_sector_pitch_rad = float(sector_pitch_rad)
        if sector_roll_rad is not None:
            self.tx_sector_roll_rad = float(sector_roll_rad)

        self.scene.frequency = float(self.fc)
        self.scene.bandwidth = float(bandwidth)
        self.scene.synthetic_array = True
        self._prepare_bs_sector_state()

    def build_standard_arrays(
        self,
        *,
        tx_rows=8,
        tx_cols=8,
        tn_rx_rows=1,
        tn_rx_cols=1,
        ntn_rows=1,
        ntn_cols=1,
    ):
        """Build the default BS/TN/NTN arrays used by the SINR workflows."""
        return {
            "bs": self._make_planar_array(
                num_rows=tx_rows,
                num_cols=tx_cols,
                pattern="tr38901",
                polarization="V",
            ),
            "tn": self._make_planar_array(
                num_rows=tn_rx_rows,
                num_cols=tn_rx_cols,
                pattern="dipole",
                polarization="V",
            ),
            "ntn": self._make_planar_array(
                num_rows=ntn_rows,
                num_cols=ntn_cols,
                pattern="vsat",
                polarization="V",
            ),
        }

    def _add_bs_sector_receivers(self, name_prefix="bs-rx"):
        if self.tx_pos is None or self.tx_pos.shape[0] == 0:
            raise ValueError("tx_pos is empty; run compute_positions first.")
        if self.nsect is None:
            raise ValueError("nsect is not set; configure sector metadata first.")

        if self.tx_orientation_rad is None:
            self._prepare_bs_sector_state()
            orientations = np.asarray(self.tx_orientation_rad, dtype=float)
        else:
            orientations = np.asarray(self.tx_orientation_rad, dtype=float)

        rx_name_list = []
        flat_idx = 0
        for bs_idx in range(self.tx_pos.shape[0]):
            for sec_idx in range(self.nsect):
                rx = self._make_receiver(
                    name=f"{name_prefix}-{bs_idx}-{sec_idx}",
                    position=self.tx_pos[bs_idx],
                    color=[1.0, 0.0, 0.0],
                    orientation=orientations[flat_idx].tolist(),
                )
                self.scene.add(rx)
                rx_name_list.append(rx.name)
                flat_idx += 1
        return rx_name_list

    def _add_bs_sector_transmitters(self, *, power_dbm, name_prefix="tx"):
        if self.tx_pos is None or self.tx_pos.shape[0] == 0:
            raise ValueError("tx_pos is empty; run compute_positions first.")
        if self.nsect is None:
            raise ValueError("nsect is not set; configure sector metadata first.")

        if self.tx_orientation_rad is None:
            self._prepare_bs_sector_state()

        tx_name_list = []
        flat_idx = 0
        for bs_idx in range(self.tx_pos.shape[0]):
            for sec_idx in range(self.nsect):
                tx = Transmitter(
                    name=f"{name_prefix}-{bs_idx}-{sec_idx}",
                    position=self.tx_pos[bs_idx],
                    power_dbm=float(power_dbm),
                    orientation=np.asarray(self.tx_orientation_rad[flat_idx], dtype=float).tolist(),
                )
                self.scene.add(tx)
                tx_name_list.append(tx.name)
                flat_idx += 1
        return tx_name_list

    def _add_tn_transmitters(self, *, power_dbm, name_prefix="tn-tx", indices=None):
        if self.tn_pos is None:
            raise ValueError("tn_pos is empty; run compute_positions first.")
        if self.tx_pos is None or self.tx_pos.shape[0] == 0:
            raise ValueError("tx_pos is empty; run compute_positions first.")

        if self.tn_bs_index is None:
            diffs = self.tn_pos[:, None, :2] - self.tx_pos[None, :, :2]
            d2 = np.sum(diffs**2, axis=2)
            tn_bs_index = np.argmin(d2, axis=1)
        else:
            tn_bs_index = np.asarray(self.tn_bs_index, dtype=int)

        if indices is None:
            indices = np.arange(self.tn_pos.shape[0], dtype=int)
        else:
            indices = np.asarray(indices, dtype=int)

        tx_name_list = []
        for i in indices:
            pos = self.tn_pos[int(i)]
            tx = Transmitter(
                name=f"{name_prefix}-{i}",
                position=pos,
                power_dbm=float(power_dbm),
            )
            self.scene.add(tx)
            if hasattr(tx, "look_at"):
                try:
                    tx.look_at(self.tx_pos[int(tn_bs_index[i])])
                except Exception:
                    pass
            tx_name_list.append(tx.name)
        return tx_name_list

    def _add_tn_receivers(self, name_prefix="tn-rx", indices=None):
        if self.tn_pos is None:
            raise ValueError("tn_pos is empty; run compute_positions first.")
        if self.tx_pos is None or self.tx_pos.shape[0] == 0:
            raise ValueError("tx_pos is empty; run compute_positions first.")

        if self.tn_bs_index is None:
            diffs = self.tn_pos[:, None, :2] - self.tx_pos[None, :, :2]
            d2 = np.sum(diffs**2, axis=2)
            tn_bs_index = np.argmin(d2, axis=1)
        else:
            tn_bs_index = np.asarray(self.tn_bs_index, dtype=int)

        if indices is None:
            indices = np.arange(self.tn_pos.shape[0], dtype=int)
        else:
            indices = np.asarray(indices, dtype=int)

        rx_name_list = []
        for i in indices:
            pos = self.tn_pos[int(i)]
            rx = self._make_receiver(
                name=f"{name_prefix}-{i}",
                position=pos,
                color=[0.0, 1.0, 0.0],
            )
            self.scene.add(rx)
            if hasattr(rx, "look_at"):
                try:
                    rx.look_at(self.tx_pos[int(tn_bs_index[i])])
                except Exception:
                    pass
            rx_name_list.append(rx.name)
        return rx_name_list

    def _add_ntn_transmitters(self, *, power_dbm, name_prefix="ntn-tx", indices=None):
        if self.rx_ntn_pos is None:
            raise ValueError("rx_ntn_pos is empty; run compute_positions first.")

        if indices is None:
            indices = np.arange(self.rx_ntn_pos.shape[0], dtype=int)
        else:
            indices = np.asarray(indices, dtype=int)

        look_target = (
            np.asarray(self.ntn_look_pos, dtype=float).tolist()
            if self.ntn_look_pos is not None
            else [0.0, 0.0, 20000.0]
        )
        tx_name_list = []
        for i in indices:
            pos = self.rx_ntn_pos[int(i)]
            tx = Transmitter(
                name=f"{name_prefix}-{i}",
                position=pos,
                power_dbm=float(power_dbm),
            )
            self.scene.add(tx)
            if hasattr(tx, "look_at"):
                try:
                    tx.look_at(look_target)
                except Exception:
                    pass
            tx_name_list.append(tx.name)
        return tx_name_list

    def _snap_to_grid(self, x, y, height_roof=None, height_ground=None):
        x = np.asarray(x)
        y = np.asarray(y)
        ix = np.searchsorted(self.cm.x, x)
        iy = np.searchsorted(self.cm.y, y)
        ix = np.clip(ix, 0, len(self.cm.x) - 1)
        iy = np.clip(iy, 0, len(self.cm.y) - 1)

        if height_roof is None:
            height_roof = self.BS_height_above_roof
        if height_ground is None:
            height_ground = self.BS_height_above_ground

        xg = self.cm.x[ix]
        yg = self.cm.y[iy]
        z = np.where(
            self.cm.bldg_grid[iy, ix],
            self.cm.zmax_grid[iy, ix] + height_roof,
            self.cm.zmin_grid[iy, ix] + height_ground
        )
        return xg, yg, z

    def _positions_from_indices(self, indices, height_roof, height_ground):
        x = self.cm.x[indices[:, 1]]
        y = self.cm.y[indices[:, 0]]
        z = np.where(
            self.cm.bldg_grid[indices[:, 0], indices[:, 1]],
            self.cm.zmax_grid[indices[:, 0], indices[:, 1]] + height_roof,
            self.cm.zmin_grid[indices[:, 0], indices[:, 1]] + height_ground
        )
        return np.column_stack((x, y, z))

    def compute_cir(
        self,
        *,
        tx_array,
        rx_array,
        add_tx_fn,
        add_rx_fn,
        max_depth=3,
        bandwidth=None,
        tx_indices=None,
        tx_batch_size=None,
        add_tx_batch_fn=None,
    ):
        """Generic CIR computation for a configured TX/RX array pair."""
        if bandwidth is not None:
            self.scene.bandwidth = bandwidth
        self.scene.synthetic_array = True
        self._clear_radio_nodes()
        self.scene.tx_array = tx_array
        self.scene.rx_array = rx_array

        p_solver = PathSolver()
        if tx_batch_size is not None:
            try:
                if tx_indices is None or add_tx_batch_fn is None:
                    raise ValueError("Batched CIR computation requires tx_indices and add_tx_batch_fn.")
                return self._solve_cir_for_tx_batches(
                    p_solver=p_solver,
                    max_depth=max_depth,
                    tx_indices=tx_indices,
                    tx_batch_size=tx_batch_size,
                    add_tx_batch_fn=add_tx_batch_fn,
                    add_rx_fn=add_rx_fn,
                )
            finally:
                self._clear_radio_nodes()
                self._best_effort_rt_memory_cleanup()

        add_tx_fn()
        add_rx_fn()
        try:
            return self._solve_cir_numpy(p_solver, max_depth=max_depth)
        finally:
            self._clear_radio_nodes()
            self._best_effort_rt_memory_cleanup()

    def compute_two_mode_cirs(
        self,
        *,
        nsect=3,
        fc=None,
        tx_rows=8,
        tx_cols=8,
        tn_rx_rows=1,
        tn_rx_cols=1,
        max_depth=3,
        bandwidth=100e6,
        tx_power_dbm=30,
        tn_tx_power_dbm=None,
        ntn_tx_power_dbm=None,
        ntn_tx_batch_size=32,
        sector_yaw_offset_rad=None,
        sector_pitch_rad=None,
        sector_roll_rad=None,
    ):
        """Compute the four CIR tensors used by the two-mode SINR workflow."""
        bs_tx_power_dbm = float(tx_power_dbm)
        if tn_tx_power_dbm is None:
            tn_tx_power_dbm = bs_tx_power_dbm
        if ntn_tx_power_dbm is None:
            ntn_tx_power_dbm = bs_tx_power_dbm

        self.configure_cir_scene(
            nsect=nsect,
            fc=fc,
            bandwidth=bandwidth,
            sector_yaw_offset_rad=sector_yaw_offset_rad,
            sector_pitch_rad=sector_pitch_rad,
            sector_roll_rad=sector_roll_rad,
        )
        arrays = self.build_standard_arrays(
            tx_rows=tx_rows,
            tx_cols=tx_cols,
            tn_rx_rows=tn_rx_rows,
            tn_rx_cols=tn_rx_cols,
        )

        a_bs_to_tn, tau_bs_to_tn = self.compute_cir(
            tx_array=arrays["bs"],
            rx_array=arrays["tn"],
            add_tx_fn=lambda: self._add_bs_sector_transmitters(
                power_dbm=bs_tx_power_dbm,
                name_prefix="tx",
            ),
            add_rx_fn=lambda: self._add_tn_receivers(name_prefix="tn"),
            max_depth=max_depth,
            bandwidth=bandwidth,
        )

        a_tn_to_bs, tau_tn_to_bs = self.compute_cir(
            tx_array=arrays["tn"],
            rx_array=arrays["bs"],
            add_tx_fn=lambda: self._add_tn_transmitters(power_dbm=tn_tx_power_dbm),
            add_rx_fn=lambda: self._add_bs_sector_receivers(name_prefix="bs-rx-ul"),
            max_depth=max_depth,
            bandwidth=bandwidth,
        )

        if self.ntn_rx is None or int(self.ntn_rx) <= 0 or self.rx_ntn_pos is None:
            a_ntn_to_bs = None
            tau_ntn_to_bs = None
            a_ntn_to_tn = None
            tau_ntn_to_tn = None
        else:
            tx_indices = np.arange(self.rx_ntn_pos.shape[0], dtype=int)
            a_ntn_to_bs, tau_ntn_to_bs = self.compute_cir(
                tx_array=arrays["ntn"],
                rx_array=arrays["bs"],
                add_tx_fn=lambda: None,
                add_rx_fn=lambda: self._add_bs_sector_receivers(name_prefix="bs-rx-int"),
                max_depth=max_depth,
                bandwidth=bandwidth,
                tx_indices=tx_indices,
                tx_batch_size=ntn_tx_batch_size,
                add_tx_batch_fn=lambda batch: self._add_ntn_transmitters(
                    power_dbm=ntn_tx_power_dbm,
                    indices=batch,
                ),
            )
            a_ntn_to_tn, tau_ntn_to_tn = self.compute_cir(
                tx_array=arrays["ntn"],
                rx_array=arrays["tn"],
                add_tx_fn=lambda: None,
                add_rx_fn=lambda: self._add_tn_receivers(name_prefix="tn-rx-int"),
                max_depth=max_depth,
                bandwidth=bandwidth,
                tx_indices=tx_indices,
                tx_batch_size=ntn_tx_batch_size,
                add_tx_batch_fn=lambda batch: self._add_ntn_transmitters(
                    power_dbm=ntn_tx_power_dbm,
                    indices=batch,
                ),
            )

        cir_out = {
            "a_bs_to_tn": a_bs_to_tn,
            "tau_bs_to_tn": tau_bs_to_tn,
            "a_tn_to_bs": a_tn_to_bs,
            "tau_tn_to_bs": tau_tn_to_bs,
            "a_ntn_to_bs": a_ntn_to_bs,
            "tau_ntn_to_bs": tau_ntn_to_bs,
            "a_ntn_to_tn": a_ntn_to_tn,
            "tau_ntn_to_tn": tau_ntn_to_tn,
        }
        for key, value in cir_out.items():
            setattr(self, key, value)
        return cir_out

    def compute_positions(self,
                        ntn_rx,
                        tn_rx,
                        azimuth,
                        elevation,
                        centerBS=True,
                        bs_boundary=0.0,
                        bs_layout="random",
                        bs_grid=None,
                        nbs=None,
                        tn_distance=1500.0,
                        tn_building_ratio="sector",
                        ntn_building_ratio=None,
                        show_xy=False,
                        plot_grid=False,
                        plot_bs=False,
                        plot_tn=False,
                        plot_ntn=False):
        """
        1) Build coverage map and determine bounding box
        2) Select random positions for TX, TN/NTN receivers
        3) Optionally place TX at (x=0, y=0) if centerBS=True
        4) Filter TN receivers by distance constraints
        5) Compute random satellite direction & project
        """

        if isinstance(tn_building_ratio, str):
            tn_building_ratio = tn_building_ratio.strip().lower()
        if isinstance(bs_layout, str):
            bs_layout = bs_layout.strip().lower()

        # 1) Create/reuse coverage map
        self.ntn_rx = ntn_rx
        if self.cm is None or self.point_type is None or self.extent is None:
            self.build_coverage_map(show_xy=show_xy, plot=False)
        else:
            if show_xy:
                x_min, x_max = self.cm.x[0], self.cm.x[-1]
                y_min, y_max = self.cm.y[0], self.cm.y[-1]
                print(f"x range: [{x_min:.3f}, {x_max:.3f}]")
                print(f"y range: [{y_min:.3f}, {y_max:.3f}]")
        x_min, x_max = self.cm.x[0], self.cm.x[-1]
        y_min, y_max = self.cm.y[0], self.cm.y[-1]

        # 2) Place a single gNB TX position on building roof
        # locations_building = np.argwhere(self.cm.bldg_grid & self.cm.in_region)        
        locations_building = np.argwhere(self.cm.bldg_grid)
        locations_outdoor = np.argwhere(~self.cm.bldg_grid)
        
        # if len(locations_building) < self.nbs:
        #     raise ValueError("Not enough building points to place the TX.")

        # If centerBS=True, force a single BS at (0,0) regardless of bs_grid
        if centerBS:
            self.nbs = 1
        elif bs_grid is not None:
            nx, ny = bs_grid
            if nx <= 0 or ny <= 0:
                raise ValueError("bs_grid must be positive, e.g. (2, 2).")
            self.nbs = int(nx * ny)
        elif nbs is not None:
            self.nbs = int(nbs)
        elif self.nbs is None:
            raise ValueError("nbs is required when bs_grid is not set.")

        # If centerBS = True and only 1 BS, force TX at (0, 0).
        if centerBS and self.nbs == 1:
            tx_x, tx_y, tx_z = self._snap_to_grid(np.array([0.0]), np.array([0.0]))
        elif bs_grid is not None:
            x_start = x_min + bs_boundary
            x_end = x_max - bs_boundary
            y_start = y_min + bs_boundary
            y_end = y_max - bs_boundary
            if x_end <= x_start or y_end <= y_start:
                raise ValueError("bs_boundary too large: no valid x/y range.")
            xs = np.linspace(x_start, x_end, nx)
            ys = np.linspace(y_start, y_end, ny)
            xx, yy = np.meshgrid(xs, ys)
            tx_x = xx.ravel()
            tx_y = yy.ravel()
            tx_x, tx_y, tx_z = self._snap_to_grid(tx_x, tx_y)
        elif bs_layout == "line":
            # Place BSs evenly on x-axis within boundary, y=0
            x_start = x_min + bs_boundary
            x_end = x_max - bs_boundary
            if x_end <= x_start:
                raise ValueError("bs_boundary too large: no valid x range.")
            tx_x = np.linspace(x_start, x_end, self.nbs)
            tx_y = np.zeros_like(tx_x)
            tx_x, tx_y, tx_z = self._snap_to_grid(tx_x, tx_y)
        else:
            if locations_outdoor.shape[0] > 0:
                tx_candidates = locations_outdoor
            elif locations_building.shape[0] > 0:
                tx_candidates = locations_building
            else:
                raise ValueError("No valid grid points available to place BS.")

            replace = tx_candidates.shape[0] < self.nbs
            tx_ind = tx_candidates[
                np.random.choice(tx_candidates.shape[0], self.nbs, replace=replace)
            ]
            tx_x = self.cm.x[tx_ind[:,1]]
            tx_y = self.cm.y[tx_ind[:,0]]
            tx_z = np.where(
                self.cm.bldg_grid[tx_ind[:, 0], tx_ind[:, 1]],
                self.cm.zmax_grid[tx_ind[:, 0], tx_ind[:, 1]] + self.BS_height_above_roof,
                self.cm.zmin_grid[tx_ind[:, 0], tx_ind[:, 1]] + self.BS_height_above_ground
            )
        self.tx_pos = np.column_stack((tx_x, tx_y, tx_z))


        # 3) Place NTN receivers
        if ntn_building_ratio is None:
            num_ntn_building = None
            num_ntn_outdoor = None
        else:
            num_ntn_building = int(round(ntn_building_ratio * ntn_rx))
            num_ntn_outdoor = ntn_rx - num_ntn_building

        if ntn_building_ratio is None:
            # Pure random from all points
            all_idx = np.argwhere(self.cm.bldg_grid | (~self.cm.bldg_grid))
            if ntn_rx > all_idx.shape[0]:
                candidate_indices = all_idx[
                    np.random.choice(all_idx.shape[0], ntn_rx, replace=True)
                ]
            else:
                candidate_indices = all_idx[
                    np.random.choice(all_idx.shape[0], ntn_rx, replace=False)
                ]
        else:
            # Random with building/outdoor ratio
            rx_ntn_building_ind = locations_building[
                np.random.choice(
                    locations_building.shape[0],
                    num_ntn_building,
                    replace=num_ntn_building > locations_building.shape[0]
                )
            ] if num_ntn_building > 0 else np.empty((0, 2), dtype=int)
            rx_ntn_outdoor_ind = locations_outdoor[
                np.random.choice(
                    locations_outdoor.shape[0],
                    num_ntn_outdoor,
                    replace=num_ntn_outdoor > locations_outdoor.shape[0]
                )
            ] if num_ntn_outdoor > 0 else np.empty((0, 2), dtype=int)
            candidate_indices = np.vstack((rx_ntn_building_ind, rx_ntn_outdoor_ind))

        def filter_indices(indices):
            # Map coverage-map indices to x/y coordinates
            rx_ntn_x = self.cm.x[indices[:, 1]]
            rx_ntn_y = self.cm.y[indices[:, 0]]
            # Filter condition: exclude points in the central region
            # Example: exclude points with |x| < 250 and |y| < 800
            mask = ~((np.abs(rx_ntn_x) < 250) & (np.abs(rx_ntn_y) < 800))
            return indices[mask]

        # Initial filtering
        filtered_indices = filter_indices(candidate_indices)

        # Keep sampling until we have enough NTN receivers
        while filtered_indices.shape[0] < ntn_rx:
            # Re-sample extra candidate points using the desired ratio
            # Use replace=True to avoid running out of candidates, then de-duplicate later
            extra_building = locations_building[
                np.random.choice(locations_building.shape[0], max(1, int(0.8 * (ntn_rx - filtered_indices.shape[0])),), replace=True)
            ]
            extra_outdoor = locations_outdoor[
                np.random.choice(locations_outdoor.shape[0], max(1, int(0.2 * (ntn_rx - filtered_indices.shape[0])),), replace=True)
            ]
            extra_candidates = np.vstack((extra_building, extra_outdoor))
            extra_filtered = filter_indices(extra_candidates)
            # Merge and de-duplicate
            filtered_indices = np.vstack((filtered_indices, extra_filtered))
            # Indices are integer arrays, so np.unique works well here
            filtered_indices = np.unique(filtered_indices, axis=0)

        # Keep only the first ntn_rx indices if we have extras
        if filtered_indices.shape[0] > ntn_rx:
            filtered_indices = filtered_indices[:ntn_rx]

        # Convert the final indices back to x/y coordinates
        rx_ntn_x = self.cm.x[filtered_indices[:, 1]]
        rx_ntn_y = self.cm.y[filtered_indices[:, 0]]
        rx_ntn_z = np.where(
            self.cm.bldg_grid[filtered_indices[:, 0], filtered_indices[:, 1]],
            self.cm.zmax_grid[filtered_indices[:, 0], filtered_indices[:, 1]] + self.ntn_height_above_roof,
            self.cm.zmin_grid[filtered_indices[:, 0], filtered_indices[:, 1]] + self.ntn_height_above_ground
        )
        self.rx_ntn_pos = np.column_stack((rx_ntn_x, rx_ntn_y, rx_ntn_z))
        
        
        # 4) Place TN receivers
        if tn_building_ratio == "sector":
            # For each BS, place 3 TN at 0/120/240 deg, distance tn_distance
            angles_deg = np.array([0.0, 120.0, 240.0])
            angles_rad = np.deg2rad(angles_deg)
            offsets = np.stack([np.cos(angles_rad), np.sin(angles_rad)], axis=1) * tn_distance
            bs_xy = self.tx_pos[:, :2]
            tn_xy = (bs_xy[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
            tn_x = np.clip(tn_xy[:, 0], x_min, x_max)
            tn_y = np.clip(tn_xy[:, 1], y_min, y_max)
            tn_x, tn_y, tn_z = self._snap_to_grid(
                tn_x,
                tn_y,
                height_roof=self.tn_height_above_roof,
                height_ground=self.tn_height_above_ground,
            )
            self.tn_pos = np.column_stack((tn_x, tn_y, tn_z))
        elif tn_building_ratio is None:
            # Random TN from all grid points
            if tn_rx <= 0:
                self.tn_pos = np.empty((0, 3))
            else:
                all_idx = np.argwhere(self.cm.bldg_grid | (~self.cm.bldg_grid))
                if tn_rx > all_idx.shape[0]:
                    chosen = all_idx[np.random.choice(all_idx.shape[0], tn_rx, replace=True)]
                else:
                    chosen = all_idx[np.random.choice(all_idx.shape[0], tn_rx, replace=False)]
                self.tn_pos = self._positions_from_indices(
                    chosen, self.tn_height_above_roof, self.tn_height_above_ground
                )
        else:
            # Random TN with building ratio
            if tn_rx <= 0:
                self.tn_pos = np.empty((0, 3))
            else:
                tn_building = int(round(float(tn_building_ratio) * tn_rx))
                tn_outdoor = tn_rx - tn_building
                bldg_idx = np.argwhere(self.cm.bldg_grid)
                out_idx = np.argwhere(~self.cm.bldg_grid)
                bldg_sel = bldg_idx[
                    np.random.choice(bldg_idx.shape[0], tn_building, replace=tn_building > bldg_idx.shape[0])
                ] if tn_building > 0 else np.empty((0, 2), dtype=int)
                out_sel = out_idx[
                    np.random.choice(out_idx.shape[0], tn_outdoor, replace=tn_outdoor > out_idx.shape[0])
                ] if tn_outdoor > 0 else np.empty((0, 2), dtype=int)
                chosen = np.vstack((bldg_sel, out_sel)) if tn_rx > 0 else np.empty((0, 2), dtype=int)
                self.tn_pos = self._positions_from_indices(
                    chosen, self.tn_height_above_roof, self.tn_height_above_ground
                )

        # 5) Compute random satellite direction & project it to bounding box
        # azimuth = np.random.uniform(0, 360)
        # elevation = np.random.uniform(25, 90)
        x_proj, y_proj, z_proj = satellite_projection(
            azimuth,
            elevation,
            self.sat_distance,
            self.L_NS,
            self.W_WE
        )
        self.ntn_look_pos = np.array([x_proj, y_proj, z_proj])

        if plot_grid or plot_bs or plot_tn or plot_ntn:
            self._plot_grid(show_bs=plot_bs, show_tn=plot_tn, show_ntn=plot_ntn)
        
        
        
        
        
