import argparse
import pathlib

import dgl
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing.pool import Pool
from itertools import repeat
import signal

try:
    from OCC.Core.STEPControl_Writer import STEPControl_Writer
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_MakeShell,
        BRepBuilderAPI_MakeSolid,
    )
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Solid
    from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline
    from OCC.Core.Geom import Geom_BSplineSurface, Geom_BSplineCurve
    from OCC.Core.gp import gp_Pnt, gp_Vec
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_C0
    HAS_OCC = True
except ImportError:
    HAS_OCC = False
    print("Warning: pythonOCC not available. Install with: conda install -c conda-forge pythonocc-core")


def reconstruct_face_from_uvgrid(face_feat, num_u_samples, num_v_samples):
    """
    从UV-grid特征重建面
    
    使用B-spline曲面拟合从采样点重建面。
    注意：这是一个近似方法，可能无法完全重建原始NURBS曲面。
    
    Args:
        face_feat: numpy array of shape (num_u_samples, num_v_samples, 7)
                   [points(3), normals(3), mask(1)]
        num_u_samples: U方向的采样点数
        num_v_samples: V方向的采样点数
    
    Returns:
        TopoDS_Face or None
    """
    if not HAS_OCC:
        return None
    
    try:
        # 提取点坐标
        points = face_feat[:, :, :3]  # (num_u_samples, num_v_samples, 3)
        mask = face_feat[:, :, 6:7].squeeze()  # (num_u_samples, num_v_samples)
        
        # 组织点为网格结构（保持UV网格结构）
        grid_points = []
        for i in range(num_u_samples):
            row = []
            for j in range(num_v_samples):
                if mask[i, j]:
                    row.append(gp_Pnt(*points[i, j]))
                else:
                    row.append(None)
            grid_points.append(row)
        
        # 检查是否有足够的有效点
        valid_count = sum(1 for row in grid_points for p in row if p is not None)
        if valid_count < 4:
            return None
        
        # 使用GeomAPI_PointsToBSplineSurface创建B-spline曲面
        # 注意：这需要将点组织成网格结构
        try:
            # 创建点数组（只包含有效点）
            point_array = []
            for i in range(num_u_samples):
                for j in range(num_v_samples):
                    if grid_points[i][j] is not None:
                        point_array.append(grid_points[i][j])
            
            if len(point_array) < 4:
                return None
            
            # 使用B-spline曲面拟合
            # 注意：GeomAPI_PointsToBSplineSurface需要点按网格顺序排列
            # 这里使用简化的方法：尝试从点云创建近似曲面
            from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
            from OCC.Core.TColgp import TColgp_Array2OfPnt
            
            # 创建点网格（使用有效点的边界框）
            # 这是一个简化的方法，实际应该保持原始UV参数化
            array2d = TColgp_Array2OfPnt(1, min(num_u_samples, 10), 1, min(num_v_samples, 10))
            
            # 填充点（简化：只使用部分点）
            u_idx, v_idx = 1, 1
            for i in range(0, num_u_samples, max(1, num_u_samples // 10)):
                for j in range(0, num_v_samples, max(1, num_v_samples // 10)):
                    if grid_points[i][j] is not None and u_idx <= 10 and v_idx <= 10:
                        array2d.SetValue(u_idx, v_idx, grid_points[i][j])
                        v_idx += 1
                    if v_idx > 10:
                        break
                if u_idx > 10:
                    break
                u_idx += 1
                v_idx = 1
            
            # 创建B-spline曲面
            bspline_builder = GeomAPI_PointsToBSplineSurface(array2d)
            if bspline_builder.IsDone():
                bspline_surface = bspline_builder.Surface()
                # 创建face
                face_maker = BRepBuilderAPI_MakeFace(bspline_surface, 1e-6)
                if face_maker.IsDone():
                    return face_maker.Face()
        except Exception as e:
            # 如果B-spline拟合失败，返回None
            # print(f"B-spline surface fitting failed: {e}")
            return None
        
        return None
    except Exception as e:
        # print(f"Error reconstructing face: {e}")
        return None


def reconstruct_edge_from_ugrid(edge_feat, num_u_samples):
    """
    从U-grid特征重建边
    
    使用B-spline曲线拟合从采样点重建边。
    注意：这是一个近似方法，可能无法完全重建原始曲线。
    
    Args:
        edge_feat: numpy array of shape (num_u_samples, 6)
                   [points(3), tangents(3)]
        num_u_samples: U方向的采样点数
    
    Returns:
        TopoDS_Edge or None
    """
    if not HAS_OCC:
        return None
    
    try:
        # 提取点坐标和切向量
        points = edge_feat[:, :3]  # (num_u_samples, 3)
        tangents = edge_feat[:, 3:6]  # (num_u_samples, 3)
        
        if len(points) < 2:
            return None
        
        # 创建点列表
        occ_points = [gp_Pnt(*p) for p in points]
        
        if len(occ_points) < 2:
            return None
        
        try:
            # 使用B-spline曲线拟合
            from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
            from OCC.Core.TColgp import TColgp_Array1OfPnt
            
            # 创建点数组
            point_array = TColgp_Array1OfPnt(1, len(occ_points))
            for i, pnt in enumerate(occ_points):
                point_array.SetValue(i + 1, pnt)
            
            # 创建B-spline曲线
            bspline_builder = GeomAPI_PointsToBSpline(point_array)
            if bspline_builder.IsDone():
                bspline_curve = bspline_builder.Curve()
                # 创建edge
                edge_maker = BRepBuilderAPI_MakeEdge(bspline_curve)
                if edge_maker.IsDone():
                    return edge_maker.Edge()
        except Exception as e:
            # 如果B-spline拟合失败，尝试创建简单的直线段
            # print(f"B-spline curve fitting failed: {e}, trying simple line")
            try:
                # 创建简单的直线（连接首尾点）
                if len(occ_points) >= 2:
                    edge_maker = BRepBuilderAPI_MakeEdge(occ_points[0], occ_points[-1])
                    if edge_maker.IsDone():
                        return edge_maker.Edge()
            except:
                pass
        
        return None
    except Exception as e:
        # print(f"Error reconstructing edge: {e}")
        return None


def build_solid_from_graph(graph, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples):
    """
    从DGL graph重建solid
    
    注意：这是一个非常困难的问题，因为：
    1. UV-grid只是采样点，不是完整的NURBS曲面表示
    2. 需要从采样点重建曲面（曲面拟合）
    3. 需要重建拓扑关系（face的边界、edge的方向等）
    
    这个函数提供了一个基础框架，实际实现需要更复杂的算法。
    
    Args:
        graph: DGL graph
        curv_num_u_samples: 曲线U方向采样点数
        surf_num_u_samples: 曲面U方向采样点数
        surf_num_v_samples: 曲面V方向采样点数
    
    Returns:
        TopoDS_Solid or None
    """
    if not HAS_OCC:
        print("Error: pythonOCC not available")
        return None
    
    try:
        # 提取节点和边特征
        face_feats = graph.ndata["x"].numpy()  # (num_faces, num_u, num_v, 7)
        edge_feats = graph.edata["x"].numpy()  # (num_edges, num_u, 6)
        
        num_faces = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        print(f"Reconstructing solid from graph: {num_faces} faces, {num_edges} edges")
        
        # 重建faces
        faces = []
        for i in range(num_faces):
            face_feat = face_feats[i]  # (num_u, num_v, 7)
            face = reconstruct_face_from_uvgrid(
                face_feat, surf_num_u_samples, surf_num_v_samples
            )
            if face is not None:
                faces.append(face)
        
        # 重建edges
        edges = []
        for i in range(num_edges):
            edge_feat = edge_feats[i]  # (num_u, 6)
            edge = reconstruct_edge_from_ugrid(edge_feat, curv_num_u_samples)
            if edge is not None:
                edges.append(edge)
        
        # 构建拓扑结构
        # 注意：这里需要根据图的拓扑关系重建face之间的连接
        # 这是一个非常复杂的过程，需要：
        # 1. 确定哪些edge属于哪些face
        # 2. 确定edge的方向
        # 3. 构建wire和shell
        # 4. 构建solid
        
        # 由于这是一个非常复杂的问题，这里只提供一个框架
        # 实际实现需要更深入的B-rep重建算法
        
        print(f"Warning: Solid reconstruction is incomplete. "
              f"This is a complex problem requiring advanced surface/curve fitting algorithms.")
        print(f"Reconstructed {len(faces)} faces and {len(edges)} edges, "
              f"but topology reconstruction is not implemented.")
        
        return None  # 暂时返回None，需要完整的实现
        
    except Exception as e:
        print(f"Error building solid from graph: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_one_file(arguments):
    """处理单个graph文件"""
    fn, args = arguments
    fn_stem = fn.stem
    output_path = pathlib.Path(args.output)
    
    try:
        # 加载DGL graph
        graphs, _ = dgl.load_graphs(str(fn))
        graph = graphs[0]  # 假设只有一个graph
        
        # 重建solid
        solid = build_solid_from_graph(
            graph, args.curv_u_samples, args.surf_u_samples, args.surf_v_samples
        )
        
        if solid is not None:
            # 保存STEP文件
            step_writer = STEPControl_Writer()
            step_writer.Transfer(solid, 1)  # 1 = STEPControl_AsIs
            output_file = output_path.joinpath(fn_stem + ".step")
            status = step_writer.Write(str(output_file))
            
            if status == IFSelect_RetDone:
                print(f"Successfully saved: {output_file}")
            else:
                print(f"Failed to save: {output_file}")
        else:
            print(f"Warning: Could not reconstruct solid from {fn_stem}")
            
    except Exception as e:
        print(f"Error processing {fn_stem}: {e}")
        import traceback
        traceback.print_exc()


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process(args):
    """处理所有graph文件"""
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    
    graph_files = list(input_path.glob("*.bin"))
    
    if len(graph_files) == 0:
        print(f"No .bin files found in {input_path}")
        return
    
    print(f"Found {len(graph_files)} graph files")
    
    if args.num_processes == 1:
        # 单进程处理
        for fn in tqdm(graph_files):
            process_one_file((fn, args))
    else:
        # 多进程处理
        pool = Pool(processes=args.num_processes, initializer=initializer)
        try:
            results = list(tqdm(
                pool.imap(process_one_file, zip(graph_files, repeat(args))),
                total=len(graph_files)
            ))
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
        print(f"Processed {len(results)} files.")


def main():
    parser = argparse.ArgumentParser(
        "Convert DGL graph BIN files to STEP files"
    )
    parser.add_argument("input", type=str, help="Input folder of DGL graph BIN files")
    parser.add_argument("output", type=str, help="Output folder of STEP files")
    parser.add_argument(
        "--curv_u_samples", type=int, default=10,
        help="Number of samples on each curve (must match graph data)"
    )
    parser.add_argument(
        "--surf_u_samples", type=int, default=10,
        help="Number of samples on each surface along the u-direction (must match graph data)"
    )
    parser.add_argument(
        "--surf_v_samples", type=int, default=10,
        help="Number of samples on each surface along the v-direction (must match graph data)"
    )
    parser.add_argument(
        "--num_processes", type=int, default=1,
        help="Number of processes to use (default: 1, as reconstruction is complex)"
    )
    args = parser.parse_args()
    
    if not HAS_OCC:
        print("Error: pythonOCC is required but not available.")
        print("Please install it with: conda install -c conda-forge pythonocc-core")
        return
    
    print("=" * 70)
    print("WARNING: Graph to Solid reconstruction is a VERY COMPLEX problem.")
    print("=" * 70)
    print("Limitations:")
    print("1. UV-grid只包含采样点，不是完整的NURBS曲面表示")
    print("2. 需要从采样点重建曲面（曲面拟合问题）")
    print("3. 需要重建拓扑关系（face边界、edge方向等）")
    print("4. 当前实现仅提供基础框架，完整的solid重建尚未实现")
    print()
    print("This script provides a framework for:")
    print("- Loading DGL graph files")
    print("- Extracting face and edge features")
    print("- Basic B-spline surface/curve fitting (partial implementation)")
    print("- Full topology reconstruction is NOT implemented")
    print("=" * 70)
    print()
    
    process(args)


if __name__ == "__main__":
    main()

