## `process/` 目录代码讲解

### 一、目录概览

`process/` 目录提供数据预处理工具，将 STEP 格式的 CAD 文件转换为 UV-Net 所需格式。

```
process/
├── README.md                    # 使用说明
├── solid_to_graph.py           # ⭐ STEP → DGL 图（核心）
├── solid_to_pointcloud.py      # STEP → 点云
├── solid_to_rendermesh.py      # STEP → 渲染网格
├── visualize.py                # 可视化（occwl viewer）
└── visualize_uvgrid_graph.py   # UV-grid 可视化（matplotlib）
```

---

### 二、核心文件：`solid_to_graph.py`

**功能**：将 STEP 文件转换为 DGL 图格式（UV-Net 训练所需）

#### 1. 核心函数：`build_graph()`

```python
def build_graph(solid, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples):
    # Build face adjacency graph with B-rep entities as node and edge features
    graph = face_adjacency(solid)

    # Compute the UV-grids for faces
    graph_face_feat = []
    for face_idx in graph.nodes:
        # Get the B-rep face
        face = graph.nodes[face_idx]["face"]
        # Compute UV-grids
        points = uvgrid(
            face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        normals = uvgrid(
            face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
        # Concatenate channel-wise to form face feature tensor
        face_feat = np.concatenate((points, normals, mask), axis=-1)
        graph_face_feat.append(face_feat)
    graph_face_feat = np.asarray(graph_face_feat)

    # Compute the U-grids for edges
    graph_edge_feat = []
    for edge_idx in graph.edges:
        # Get the B-rep edge
        edge = graph.edges[edge_idx]["edge"]
        # Ignore dgenerate edges, e.g. at apex of cone
        if not edge.has_curve():
            continue
        # Compute U-grids
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        # Concatenate channel-wise to form edge feature tensor
        edge_feat = np.concatenate((points, tangents), axis=-1)
        graph_edge_feat.append(edge_feat)
    graph_edge_feat = np.asarray(graph_edge_feat)

    # Convert face-adj graph to DGL format
    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
    dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)
    return dgl_graph
```

**处理流程**：

**步骤1：构建面邻接图**

```python
graph = face_adjacency(solid)
```

- 使用 `occwl` 构建面邻接图
- 节点 = B-rep 面
- 边 = 相邻面的连接

**步骤2：为每个面生成 2D UV-grid**

```python
for face_idx in graph.nodes:
    face = graph.nodes[face_idx]["face"]
    points = uvgrid(face, method="point", ...)      # 点坐标
    normals = uvgrid(face, method="normal", ...)    # 法向量
    visibility_status = uvgrid(face, method="visibility_status", ...)  # 可见性
    mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 掩码
    face_feat = np.concatenate((points, normals, mask), axis=-1)  # (10, 10, 7)
```

**UV-grid 格式**：

- `points`: `(10, 10, 3)` - 点坐标
- `normals`: `(10, 10, 3)` - 法向量
- `mask`: `(10, 10, 1)` - 修剪掩码（1=可见，0=不可见）
- 合并：`(10, 10, 7)`

**步骤3：为每条边生成 1D UV-grid**

```python
for edge_idx in graph.edges:
    edge = graph.edges[edge_idx]["edge"]
    if not edge.has_curve():
        continue  # 跳过退化边
    points = ugrid(edge, method="point", ...)      # 点坐标
    tangents = ugrid(edge, method="tangent", ...)  # 切线方向
    edge_feat = np.concatenate((points, tangents), axis=-1)  # (10, 6)
```

**UV-grid 格式**：

- `points`: `(10, 3)` - 点坐标
- `tangents`: `(10, 3)` - 切线方向
- 合并：`(10, 6)`

**步骤4：转换为 DGL 图格式**

```python
dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)  # 节点特征
dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)   # 边特征
```

**输出格式**：

- `ndata["x"]`: `(num_faces, 10, 10, 7)` - 面 UV-grid
- `edata["x"]`: `(num_edges, 10, 6)` - 边 UV-grid

#### 2. 批量处理函数

```python
def process_one_file(arguments):
    fn, args = arguments
    fn_stem = fn.stem
    output_path = pathlib.Path(args.output)
    solid = load_step(fn)[0]  # Assume there's one solid per file
    graph = build_graph(
        solid, args.curv_u_samples, args.surf_u_samples, args.surf_v_samples
    )
    dgl.data.utils.save_graphs(str(output_path.joinpath(fn_stem + ".bin")), [graph])
```

**功能**：

- 加载 STEP 文件
- 构建 DGL 图
- 保存为 `.bin` 文件

#### 3. 多进程处理

```python
def process(args):
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    step_files = list(input_path.glob("*.st*p"))
    pool = Pool(processes=args.num_processes, initializer=initializer)
    try:
        results = list(tqdm(pool.imap(process_one_file, zip(step_files, repeat(args))), total=len(step_files)))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    print(f"Processed {len(results)} files.")
```

**特点**：

- 多进程并行处理
- 支持中断处理（Ctrl+C）
- 进度条显示

#### 4. 使用示例

```bash
python -m process.solid_to_graph \
    /path/to/step_files \
    /path/to/output \
    --curv_u_samples 10 \
    --surf_u_samples 10 \
    --surf_v_samples 10 \
    --num_processes 8
```

---

### 三、其他转换工具

#### 1. `solid_to_pointcloud.py` - 点云转换

**功能**：将 STEP 文件转换为点云格式

```python
verts, tris, tri_mapping = triangulate_with_face_mapping(solid)

mesh = trimesh.Trimesh(vertices=verts, faces=tris)
points, face_indices = trimesh.sample.sample_surface(mesh, args.num_points)
points_to_face_mapping = tri_mapping[face_indices]

np.savez(
    str(output_path.joinpath(fn_stem + ".npz")),
    points=points,
    point_mapping=points_to_face_mapping,
)
```

**输出**：

- `.npz` 文件
- `points`: 点坐标 `(N, 3)`
- `point_mapping`: 点到面的映射

**用途**：用于点云方法或可视化

#### 2. `solid_to_rendermesh.py` - 渲染网格转换

**功能**：将 STEP 文件转换为 STL 格式的渲染网格

```python
def triangulate_with_face_mapping(solid, triangle_face_tol=0.01, angle_tol_rads=0.1):
    # Triangulate faces
    solid.triangulate_all_faces(
        triangle_face_tol=triangle_face_tol, angle_tol_rads=angle_tol_rads
    )
    # ... 处理三角化 ...
    return verts, tris, tri_mapping
```

**输出**：

- `.stl` 文件（非水密网格，用于渲染）

**用途**：用于渲染和可视化

---

### 四、可视化工具

#### 1. `visualize.py` - occwl 可视化

**功能**：使用 occwl 的 Viewer 可视化 STEP 文件和 DGL 图

```python
def draw_face_uvgrids(solid, graph, viewer):
    face_uvgrids = graph.ndata["x"].view(-1, 7)
    points = []
    normals = []
    for idx in range(face_uvgrids.shape[0]):
        if face_uvgrids[idx, -1] == 0:  # 跳过不可见点
            continue
        points.append(face_uvgrids[idx, :3].cpu().numpy())
        normals.append(face_uvgrids[idx, 3:6].cpu().numpy())
    # 绘制点和法向量
    viewer.display_points(points, ...)
    # 绘制法向量箭头
    for pt, nor in zip(points, normals):
        viewer.display(Edge.make_line_from_points(...), ...)
```

**功能**：

- 绘制面 UV-grid 点
- 绘制面法向量
- 绘制边 UV-grid 点
- 绘制边切线
- 绘制面邻接图边

**使用**：

```bash
python -m process.visualize solid.step graph.bin
```

#### 2. `visualize_uvgrid_graph.py` - matplotlib 可视化

**功能**：使用 matplotlib 可视化 UV-grid 和面邻接图

```python
def plot_uvsolid(uvsolid: torch.Tensor, ax, normals=False):
    # 绘制面的 UV-grid 点
    for i in range(num_faces):
        pts = uvsolid[i, :, :, :3].cpu().detach().numpy().reshape((-1, 3))
        mask = uvsolid[i, :, :, 6].cpu().detach().numpy().reshape(-1)
        point_indices_inside_faces = mask == 1
        pts = pts[point_indices_inside_faces, :]
        if normals:
            # 绘制法向量箭头
            ax.quiver(...)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
```

**功能**：

- 绘制面 UV-grid（可选法向量）
- 绘制边 UV-grid（可选切线）
- 绘制面邻接图连接
- 保存为图片

**使用**：

```bash
python -m process.visualize_uvgrid_graph /path/to/bin_files \
    --plot_face_normals \
    --plot_edge_tangents
```

---

### 五、数据流总结

```
STEP 文件 (.stp/.step)
    ↓
solid_to_graph.py
    ├─ 构建面邻接图
    ├─ 生成面 UV-grid (10×10×7)
    ├─ 生成边 UV-grid (10×6)
    └─ 转换为 DGL 图
    ↓
DGL 图文件 (.bin)
    ├─ ndata["x"]: 面 UV-grid
    └─ edata["x"]: 边 UV-grid
    ↓
UV-Net 训练/测试
```

---

### 六、关键设计要点

1. **多进程处理**：支持并行处理多个文件
2. **错误处理**：跳过退化边和空文件
3. **可配置采样**：可调整 UV-grid 采样密度
4. **格式标准化**：统一输出 DGL `.bin` 格式
5. **可视化支持**：提供两种可视化工具

---

### 七、使用建议

1. **核心工具**：`solid_to_graph.py` 是 UV-Net 训练必需
2. **可视化**：用于调试和验证数据
3. **其他工具**：点云和网格转换用于其他用途