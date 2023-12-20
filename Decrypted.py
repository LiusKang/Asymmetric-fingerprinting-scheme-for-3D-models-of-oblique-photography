# -*- coding: utf-8 -*-
import open3d as o3d
import sys
import numpy as np
from PIL import Image  # 库的原因读成True/False
import math
import cv2
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Encrypted
import paillier
import Rotate
import Pan


# 读取OJB数据
def Get_OBJ(ori_obj, ori_picture):
    obj_mesh = o3d.io.read_triangle_mesh(ori_obj)
    obj_picture01 = Image.open(ori_picture)
    if obj_mesh is None or obj_picture01 is None:
        sys.exit('Could not open {0}.'.format(ori_obj))
    obj_pc = np.asarray(obj_mesh.vertices)  # 顶点数据数组Vector3[]
    obj_sjm = np.asarray(obj_mesh.triangles)  # 三角形顶点索引数组，int[],可能是三角网点的顺序
    uv = np.asarray(obj_mesh.triangle_uvs)  # （uv）纹理坐标数组，Vector2[]
    wenli_iamge = obj_mesh.textures
    return obj_mesh, obj_pc, obj_sjm, uv, wenli_iamge


# 生成新的模型数据
# def create_newobj(new_arr, ima_path):
#     Mesh = o3d.geometry.TriangleMesh()
#     Mesh.vertices = o3d.utility.Vector3dVector(new_arr)
#     Mesh.triangles = o3d.utility.Vector3iVector(GCI)
#     Mesh.triangle_uvs = o3d.open3d.utility.Vector2dVector(TC)
#     tex_img = cv2.imread(ima_path)
#     tex_img = cv2.cvtColor(tex_img, cv2.COLOR_BGR2RGB)
#     tex_img = cv2.flip(tex_img, 0)
#     Mesh.textures = [o3d.geometry.Image(tex_img)]
#     Mesh.compute_vertex_normals()
#     o3d.io.write_triangle_mesh("DE_W_Model.obj", Mesh)

# 数据归一化处理
def min_max_normalize(data, flage):
    R = 1e9  # 1e6
    # z坐标归一化 flage = 1
    # 投影长度归一化 flage = 其他
    if flage == 1:
        # Model_1
        # data_min = -35.811564402030356  # * 2
        # data_max = 36.46097542865367  # * 2
        # Model_2
        # data_min = -40.38008318840921  # * 2
        # data_max = 73.5308592814375  # * 2
        # Model_3
        # data_min = -24.936616850439393  # * 2
        # data_max = 32.747161292225925  # * 2
        # Model_4
        data_min = -8.876511338416655  # * 2
        data_max = 35.12811099482001  # * 2
        # Model_5
        # data_min = -43.740985825397914  # * 2
        # data_max = 69.03685288680758  # * 2
        # Model_10.3
        # data_min = -81.1380034433989  # * 2
        # data_max = 32.118315129090455  # * 2

        normalized_data = [((x - data_min) / (data_max - data_min)) * R for x in data]
        intnormalized_data = [int(round(x)) for x in normalized_data]
    else:
        data_min = min(data)
        data_max = max(data)
        normalized_data = [((x - data_min) / (data_max - data_min)) * R for x in data]
        intnormalized_data = [int(round(x)) for x in normalized_data]
    return intnormalized_data


def Dectect_fingerprint(Decrypted_W_Z, normalized_distance, FS):
    ListZ_F = len(FS) * [0]
    for i, row in zip(Decrypted_W_Z, normalized_distance):
        if i == 0 or i == 1e9:
            continue
        index_x = row % len(FS)
        thousands_digits = (i // 1000) % 10
        hundredds_digits = (i // 100) % 10
        # 奇偶相同，嵌入水印为1
        if hundredds_digits % 2 == thousands_digits % 2:
            ListZ_F[index_x] = ListZ_F[index_x] + 1
        # 奇偶不同，嵌入水印为0
        else:
            ListZ_F[index_x] = ListZ_F[index_x] - 1

    for p in range(0, len(ListZ_F)):
        if ListZ_F[p] > 0:
            ListZ_F[p] = 1
        else:
            ListZ_F[p] = 0
    return ListZ_F


def nc(FS, ListZ_F):
    s = 0
    for i in range(0, len(FS)):
        if FS[i] == ListZ_F[i]:
            s += 1
        else:
            s += 0
    nc = s / len(FS)
    return nc


def calculate_distance(point_a, point_b):
    x1, y1, z1 = point_a
    x2, y2, z2 = point_b
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return distance


if __name__ == '__main__':
    # 数据属性信息
    '''
    Number_meshes(NM)：三角网格个数 
    Vertex_coordinates(VC): 顶点坐标二维列表，[[x,y,z],[x,y,z]]
    Grid_coordinate_index(GCI): 三角网格坐标点索引
    Texture_coordinates(TC): 纹理坐标数组
    Textured_imagery(TI): 纹理影像
    '''
    NM, VC, GCI, TC, TI = Get_OBJ(r"D:\Specialized Software\PyCharm\WorkSpace\3DWaterMark\data\4_DW.ply",
                                  r"D:\Specialized Software\PyCharm\WorkSpace\3DWaterMark\data\Tile_+008_+006_0.jpg")

    # 三维模型数据
    vertices = VC.tolist()

    # 不攻击提取
    # Model_1:
    # A11 = np.array([-105.85587311, -2267.77758789, 20.97569847])
    # B11 = np.array([-107.39753723, -2269.34228516, 13.71096897])
    # C11 = np.array([-105.89181519, -2274.33398438, 23.01403427])
    # Model_2:
    # A11 = np.array([-126.16944122, -2347.88574219, 33.48313522])
    # B11 = np.array([-124.51350403, -2352.7421875, 36.02323532])
    # C11 = np.array([-124.07032776, -2354.13452148, 31.12036324])
    # Model_3:
    # A11 = np.array([-135.75205994, -2404.79370117, 34.68191147])
    # B11 = np.array([-134.75428772, -2410.02416992, 32.57728577])
    # C11 = np.array([-134.80067444, -2410.17578125, 37.13083649])
    # Model_4:
    A11 = np.array([-11.05858803, -2419.62792969, 12.66919899])
    B11 = np.array([-11.5785656, -2412.66821289, 11.6200161])
    C11 = np.array([-11.98904514, -2417.21777344, 16.21142578])
    # Model_5:
    # A11 = np.array([-1134.10571289, -115.83377838, 69.26651001])
    # B11 = np.array([-1137.17016602, -116.63397217, 46.57534027])
    # C11 = np.array([-1112.77978516, -113.53468323, 71.6571579])
    # Model_10.3
    # A11 = np.array([-135.72576904, -2405.04150391, 30.81289482])
    # B11 = np.array([-135.69628906,-2405.59448242,21.79308701])
    # C11 = np.array([-134.38943481, -2410.88110352, 25.1788559])

    ## 平移攻击矫正
    # A11 = np.array([-1134.10571289, -115.83377838, 69.26651001])
    # B11 = np.array([-1137.17016602, -116.63397217, 46.57534027])
    # C11 = np.array([-1112.77978516, -113.53468323, 71.6571579])

    ## 旋转攻击矫正
    # A11 = [-1134.10571289, -115.83377838, 69.26651001]
    # B11 = [-1137.17016602, -116.63397217, 46.57534027]
    # C11 = [-1112.77978516, -113.53468323, 71.6571579]
    #
    # # 旋转条件
    # angle = 90  # 旋转角度（以度为单位）
    # axis = 'z'  # 旋转轴
    #
    # # 旋转
    # A11 = np.array(Rotate.rotate_points(A11, angle, axis))
    # B11 = np.array(Rotate.rotate_points(B11, angle, axis))
    # C11 = np.array(Rotate.rotate_points(C11, angle, axis))

    ## 缩放攻击 同时修改归一化函数中的最大最小值
    # A111 = np.array([-1134.10571289, -115.83377838, 69.26651001])
    # B111 = np.array([-1137.17016602, -116.63397217, 46.57534027])
    # C111 = np.array([-1112.77978516, -113.53468323, 71.6571579])
    # 定义缩放比例
    # scale_factor = 2
    # 使用广播功能缩放数组
    # A11 = A111 * scale_factor
    # B11 = B111 * scale_factor
    # C11 = C111 * scale_factor

    # 计算新的坐标轴与原点
    vo, x_axis, y_axis, z_axis = Encrypted.find_coordinate_axes(A11, B11, C11)

    # 将新坐标系原点移到[0，0，0]
    result_list = [[x - y for x, y in zip(sublist, vo)] for sublist in vertices]

    # 将原始坐标转换到新的坐标系下
    transformed_vertices = Encrypted.transform_to_new_coordinates(result_list, x_axis, y_axis, z_axis)

    # 计算所有点到给原点的距离的投影长度
    distances = [np.linalg.norm([np.array(point[0]), np.array(point[1]), 0]) for point in transformed_vertices]

    # 进行最大最小归一化
    normalized_distance = min_max_normalize(distances, 0)

    ### 分块指纹提取
    # Z坐标提取与整数化
    scz = [row[2] for row in transformed_vertices]

    # 归一化
    normalized_scz = min_max_normalize(scz, 1)

    PFS = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    UFS = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]

    # 数据分块
    Block_Index = [int(str(num)[0:4]) for num in normalized_scz]  # [12,23,45,56]
    Block_Index_Bit = [num % 2 for num in Block_Index]  # [0,1,1,0]
    # 用于嵌入发行商追踪指纹
    P_normalized_scz = [normalized_scz[i] for i in range(len(normalized_scz)) if Block_Index_Bit[i] == 0]
    P_normalized_distance = [normalized_distance[i] for i in range(len(normalized_distance)) if Block_Index_Bit[i] == 0]
    # 用于嵌入用户指纹
    U_normalized_scz = [normalized_scz[i] for i in range(len(normalized_scz)) if Block_Index_Bit[i] == 1]
    U_normalized_distance = [normalized_distance[i] for i in range(len(normalized_distance)) if Block_Index_Bit[i] == 1]

    # 提取水印序列
    ListZ_PFS = Dectect_fingerprint(P_normalized_scz, P_normalized_distance, PFS)
    ListZ_UFS = Dectect_fingerprint(U_normalized_scz, U_normalized_distance, PFS)

    NCPFS = nc(PFS, ListZ_PFS)
    NCUFS = nc(UFS, ListZ_UFS)
    print(NCPFS)
    print(NCUFS)

    ### 单一指纹提取
    # # Z坐标提取与整数化
    # scz = [row[2] for row in transformed_vertices]
    # # 归一化
    # normalized_scz = min_max_normalize(scz, 1)
    #
    # FS = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
    #       0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1]
    # # 提取水印序列
    # ListZ_F = Dectect_fingerprint(normalized_scz, normalized_distance, FS)
    # NC = nc(FS, ListZ_F)
    # print(NC)