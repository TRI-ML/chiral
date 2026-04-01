# Type Reference

Python types are defined in `chiral/types.py`; C++ types are defined in `include/chiral/types.hpp`.

---

## CameraConfig

Describes one camera stream. Passed to `PolicyServer.__init__` (Python) or the `PolicyServer` constructor (C++) so the base class can pre-allocate buffers before any step is taken.

=== "Python"

    ```python
    @dataclass
    class CameraConfig:
        name:        str
        height:      int
        width:       int
        channels:    int        = 3
        has_depth:   bool       = False
        intrinsics:  np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
        extrinsics:  np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
        image_dtype: np.dtype   = np.uint8
        depth_dtype: np.dtype   = np.float32
    ```

=== "C++"

    ```cpp
    struct CameraConfig {
        std::string     name;
        int             height{0};
        int             width{0};
        int             channels{3};
        bool            has_depth{false};
        Eigen::Matrix3d intrinsics = Eigen::Matrix3d::Identity();  ///< camera matrix K
        Eigen::Matrix4d extrinsics = Eigen::Matrix4d::Identity();  ///< camera-to-world T
    };
    ```

### Fields

| Field | Python type | C++ type | Default | Description |
|-------|-------------|----------|---------|-------------|
| `name` | `str` | `string` | — | Unique camera identifier. Used as the key when looking up cameras in an `Observation`. |
| `height` | `int` | `int` | — | Image height in pixels. |
| `width` | `int` | `int` | — | Image width in pixels. |
| `channels` | `int` | `int` | `3` | Number of color channels. `3` for RGB, `1` for grayscale. |
| `has_depth` | `bool` | `bool` | `False` / `false` | Whether to allocate and stream a depth buffer for this camera. If `False`, the depth buffer is never allocated and `CameraInfo.depth` will be `None` / empty. |
| `intrinsics` | `ndarray(3,3) float64` | `Eigen::Matrix3d` | identity | Initial camera matrix K. The server base class copies this into its internal intrinsics buffer. Update per-frame with `update_intrinsics` if the focal length can change. |
| `extrinsics` | `ndarray(4,4) float64` | `Eigen::Matrix4d` | identity | Initial camera-to-world homogeneous transform T. Update per-frame with `update_extrinsics` for cameras that move (e.g. wrist cameras). |
| `image_dtype` | `np.dtype` | — | `np.uint8` | Element dtype of the pixel buffer. C++ always uses `uint8_t`. |
| `depth_dtype` | `np.dtype` | — | `np.float32` | Element dtype of the depth buffer. C++ always uses `float`. |

---

## ProprioConfig

Describes one proprioception stream. Passed alongside `CameraConfig` so the server can pre-allocate buffers.

=== "Python"

    ```python
    @dataclass
    class ProprioConfig:
        name:  str
        size:  int          # number of elements
        dtype: np.dtype = np.float32
    ```

=== "C++"

    ```cpp
    struct ProprioConfig {
        std::string name;
        int size{0};  ///< number of float32 elements
    };
    ```

### Fields

| Field | Python type | C++ type | Default | Description |
|-------|-------------|----------|---------|-------------|
| `name` | `str` | `string` | — | Unique stream identifier. Used as the key in `obs.proprios["name"]` (Python) or `obs.proprio("name")` (C++). |
| `size` | `int` | `int` | — | Number of float32 elements in the vector. Must match the actual data written via `update_proprio`. |
| `dtype` | `np.dtype` | — | `np.float32` | Element dtype. Python only; C++ always uses `float`. |

---

## CameraInfo

Contains the sensor data for one camera at a single point in time. Returned as part of an `Observation`.

=== "Python"

    ```python
    @dataclass
    class CameraInfo:
        name:       str
        intrinsics: np.ndarray          # (3, 3) float64 — camera matrix K
        extrinsics: np.ndarray          # (4, 4) float64 — camera-to-world T
        image:      np.ndarray          # (H, W, C) with dtype = image_dtype
        depth:      Optional[np.ndarray] = None  # (H, W) float32, metres; None if has_depth=False
    ```

=== "C++"

    ```cpp
    struct CameraInfo {
        std::string     name;
        Eigen::Matrix3d intrinsics = Eigen::Matrix3d::Identity();  ///< camera matrix K
        Eigen::Matrix4d extrinsics = Eigen::Matrix4d::Identity();  ///< camera-to-world T

        std::vector<uint8_t> image;           ///< raw pixel bytes
        std::vector<int>     image_shape;     ///< [H, W, C]
        std::string          image_dtype;     ///< e.g. "uint8"

        bool                 has_depth{false};
        std::vector<uint8_t> depth_data;      ///< raw depth bytes (float32 on wire)
        std::vector<int>     depth_shape;     ///< [H, W]
        std::string          depth_dtype;     ///< e.g. "float32"
    };
    ```

### Fields

| Field | Python type | C++ type | Description |
|-------|-------------|----------|-------------|
| `name` | `str` | `string` | Camera name, matching the `CameraConfig.name` on the server. |
| `intrinsics` | `ndarray(3,3) float64` | `Eigen::Matrix3d` | Camera matrix K. Sent in every observation — always current, never stale. The 3×3 matrix has layout `[[fx,0,cx],[0,fy,cy],[0,0,1]]`. |
| `extrinsics` | `ndarray(4,4) float64` | `Eigen::Matrix4d` | Camera-to-world homogeneous transform T. Sent every observation. For a moving camera (e.g. wrist), this reflects the pose at the time `make_obs()` was called. |
| `image` | `ndarray(H,W,C)` | `vector<uint8_t>` | Pixel data. In Python the shape is `(H, W, C)` with element dtype from `CameraConfig.image_dtype`. In C++ it is raw bytes of length `H*W*C`. |
| `image_shape` | — | `vector<int>` | `[H, W, C]` — C++ only; use `cam.image.shape` in Python. |
| `image_dtype` | — | `string` | Element dtype string, e.g. `"uint8"` — C++ only. |
| `depth` | `ndarray(H,W) float32` or `None` | — | Python only. Depth map in metres. `None` if `has_depth=False`. |
| `has_depth` | — | `bool` | C++ only. `true` if `depth_data` is populated. |
| `depth_data` | — | `vector<uint8_t>` | C++ only. Raw depth bytes. Reinterpret as `float*` for `H*W` float32 values. |
| `depth_shape` | — | `vector<int>` | `[H, W]` — C++ only. |
| `depth_dtype` | — | `string` | `"float32"` — C++ only. |

### Usage Examples

=== "Python"

    ```python
    cam = obs["wrist_cam"]

    # Pixel data
    image = cam.image          # np.ndarray (H, W, 3) uint8

    # Depth (check for None first)
    if cam.depth is not None:
        depth_m = cam.depth    # np.ndarray (H, W) float32, metres

    # Camera matrix — fresh every step
    K  = cam.intrinsics        # (3, 3) float64
    fx = K[0, 0]; fy = K[1, 1]
    cx = K[0, 2]; cy = K[1, 2]

    # Camera-to-world — fresh every step
    T  = cam.extrinsics        # (4, 4) float64
    R  = T[:3, :3]             # rotation (3, 3)
    t  = T[:3, 3]              # position (3,)
    ```

=== "C++"

    ```cpp
    const auto& cam = obs["wrist_cam"];

    // Pixel data as Eigen map (zero-copy)
    int H = cam.image_shape[0], W = cam.image_shape[1], C = cam.image_shape[2];
    // cam.image.data() is a uint8_t* of length H*W*C

    // Depth map (check has_depth first)
    if (cam.has_depth) {
        const float* depth_ptr =
            reinterpret_cast<const float*>(cam.depth_data.data());
        // depth_ptr[r * W + c] gives metres at row r, col c
    }

    // Camera matrix — Eigen::Matrix3d — fresh every step
    double fx = cam.intrinsics(0, 0);
    double fy = cam.intrinsics(1, 1);
    double cx = cam.intrinsics(0, 2);
    double cy = cam.intrinsics(1, 2);

    // Camera-to-world — Eigen::Matrix4d — fresh every step
    Eigen::Matrix3d R = cam.extrinsics.block<3, 3>(0, 0);
    Eigen::Vector3d t = cam.extrinsics.col(3).head<3>();
    ```

---

## ProprioInfo (C++) / proprios dict (Python)

Holds the data for one proprioception stream at a single point in time.

=== "Python"

    In Python, proprioception data is stored in a plain `dict[str, np.ndarray]` on the `Observation`. Each value is a 1-D `float32` ndarray of length `ProprioConfig.size`.

    ```python
    # Access by name
    joint_pos = obs.proprios["joint_pos"]  # np.ndarray (DOF,) float32

    # Iterate
    for name, arr in obs.proprios.items():
        print(f"{name}: {arr}")
    ```

=== "C++"

    ```cpp
    struct ProprioInfo {
        std::string        name;
        std::vector<float> data;  ///< float32 values, length = ProprioConfig.size
    };
    ```

    ```cpp
    // Access by name (throws std::out_of_range if absent)
    const chiral::ProprioInfo& jp = obs.proprio("joint_pos");
    const std::vector<float>&  q  = jp.data;   // length DOF

    // Zero-copy Eigen map
    Eigen::Map<const Eigen::VectorXf> qvec(q.data(), q.size());

    // Iterate all streams
    for (const auto& p : obs.proprios)
        std::printf("  %s[%zu]\n", p.name.c_str(), p.data.size());
    ```

---

## Observation

Snapshot of all sensor data at one point in time. Returned by `reset()` and `step()`.

=== "Python"

    ```python
    @dataclass
    class Observation:
        cameras:   list[CameraInfo]
        proprios:  dict[str, np.ndarray] = field(default_factory=dict)
        timestamp: float = 0.0
        extra:     dict  = field(default_factory=dict)

        def __getitem__(self, name: str) -> CameraInfo:
            """Look up a camera by name. Raises KeyError if not found."""
            for cam in self.cameras:
                if cam.name == name:
                    return cam
            raise KeyError(name)
    ```

=== "C++"

    ```cpp
    struct Observation {
        std::vector<CameraInfo>  cameras;
        std::vector<ProprioInfo> proprios;
        double timestamp{0.0};

        /// Look up a camera by name. Throws std::out_of_range if not found.
        CameraInfo& operator[](const std::string& name);
        const CameraInfo& operator[](const std::string& name) const;

        /// Look up a proprio stream by name. Throws std::out_of_range if not found.
        ProprioInfo& proprio(const std::string& name);
        const ProprioInfo& proprio(const std::string& name) const;
    };
    ```

### Fields

| Field | Python type | C++ type | Description |
|-------|-------------|----------|-------------|
| `cameras` | `list[CameraInfo]` | `vector<CameraInfo>` | One entry per camera declared in `camera_configs()`. Order matches declaration order. |
| `proprios` | `dict[str, ndarray]` | `vector<ProprioInfo>` | One entry per proprio stream declared in `proprio_configs()`. In Python a dict; in C++ a vector (use `obs.proprio(name)` to look up by name). |
| `timestamp` | `float` | `double` | Application-defined timestamp. Set by `_make_obs(timestamp)` / `make_obs(timestamp)` on the server. |
| `extra` | `dict` | — | Python only. Arbitrary additional data included in the observation frame header. |

---

## Action (C++ only)

Encodes a policy action to send to the server. In Python, actions are plain `np.ndarray` values passed directly to `env.step()`.

```cpp
struct Action {
    std::vector<float> data;  ///< N*D floats, row-major
    int N{0}, D{0};
};
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `data` | `vector<float>` | Action values in row-major order. Length must equal `N * D`. |
| `N` | `int` | Number of action steps (action chunk length). Use `N=1` for single-step policies. |
| `D` | `int` | Action dimensionality (e.g. 7 for a 7-DOF arm). |

### Usage

```cpp
// Single-step, 7-DOF joint position action
chiral::Action action;
action.N = 1;
action.D = 7;
action.data.assign(7, 0.f);   // zero action

// Multi-step chunk (e.g. N=8 for diffusion policy)
chiral::Action chunk;
chunk.N = 8;
chunk.D = 7;
chunk.data.assign(8 * 7, 0.f);

// From an Eigen matrix (any rows×cols float matrix)
Eigen::Matrix<float, 1, 7> mat = Eigen::Matrix<float, 1, 7>::Zero();
chiral::Action a;
a.N = 1; a.D = 7;
a.data.assign(mat.data(), mat.data() + mat.size());
```

---

## StepResult (C++ only)

Returned by `PolicyClient::step()`. In Python, `env.step()` returns a plain 5-tuple `(obs, reward, terminated, truncated, info)`.

```cpp
struct StepResult {
    Observation obs;
    float       reward{0.f};
    bool        terminated{false};
    bool        truncated{false};
    InfoMap     info;
};
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `obs` | `Observation` | Full sensor observation for this step. |
| `reward` | `float` | Scalar reward returned by the server's `step()` implementation. |
| `terminated` | `bool` | `true` if the episode ended naturally (task success or failure). |
| `truncated` | `bool` | `true` if the episode was cut short by a time limit or external condition. |
| `info` | `InfoMap` | Extra key-value pairs from the server's `step()` return value. All values are strings. |

### Usage

```cpp
auto res = env.step(action);

obs           = std::move(res.obs);
float reward  = res.reward;
bool done     = res.terminated || res.truncated;

if (done) {
    std::printf("Episode ended: terminated=%d  truncated=%d\n",
                res.terminated, res.truncated);
    break;
}
```

---

## InfoMap (C++ only)

```cpp
using InfoMap = std::unordered_map<std::string, std::string>;
```

`InfoMap` is the type used for metadata and step info in the C++ API. It maps string keys to string values. All non-string metadata from the server (integers, floats, lists) is stringified before insertion — use `std::stoi`, `std::stof`, or a custom parser to recover the original type.

!!! note "Python equivalents"
    In Python, the corresponding types are plain `dict` (for metadata and step info). Python dicts can hold arbitrary msgpack-compatible values (strings, numbers, lists, nested dicts).

### Usage

```cpp
// Reading metadata
chiral::InfoMap meta = env.get_metadata();

// Safe access with fallback
std::string cam_list = meta.count("cameras") ? meta.at("cameras") : "";

// Parse a numeric value
int action_D = meta.count("action_D") ? std::stoi(meta.at("action_D")) : 7;

// Server side: building a metadata response
chiral::InfoMap get_metadata() override {
    return {
        {"cameras",      "wrist_cam,head_cam"},
        {"action_N",     "1"},
        {"action_D",     "7"},
        {"control_hz",   "20"},
        {"robot",        "franka_panda"},
    };
}
```
