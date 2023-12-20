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
import paillier
import Decrypted


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
def create_newobj(new_arr, ima_path):
    Mesh = o3d.geometry.TriangleMesh()
    Mesh.vertices = o3d.utility.Vector3dVector(new_arr)
    Mesh.triangles = o3d.utility.Vector3iVector(GCI)
    Mesh.triangle_uvs = o3d.open3d.utility.Vector2dVector(TC)
    tex_img = cv2.imread(ima_path)
    tex_img = cv2.cvtColor(tex_img, cv2.COLOR_BGR2RGB)
    tex_img = cv2.flip(tex_img, 0)
    Mesh.textures = [o3d.geometry.Image(tex_img)]
    Mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(r"D:\Specialized Software\PyCharm\WorkSpace\3DWaterMark\data\4_DW.ply", Mesh)


# 输出最大三角网格及三个顶点坐标
def tri_area(VC, GCI):
    area_all = []
    for i in range(GCI.shape[0]):
        A, B, C = VC[GCI[i][0]], VC[GCI[i][1]], VC[GCI[i][2]]

        # 计算向量AB和向量AC
        AB = (B[0] - A[0], B[1] - A[1], B[2] - A[2])
        AC = (C[0] - A[0], C[1] - A[1], C[2] - A[2])

        # 计算向量AB和向量AC的叉积
        cross_product = (AB[1] * AC[2] - AB[2] * AC[1], AB[2] * AC[0] - AB[0] * AC[2], AB[0] * AC[1] - AB[1] * AC[0])

        # 计算叉积的模长
        cross_product_length = math.sqrt(cross_product[0] ** 2 + cross_product[1] ** 2 + cross_product[2] ** 2)

        # 计算三角形面积
        triangle_area = 0.5 * cross_product_length

        area_all.append(triangle_area)

        Max_area = max(area_all)
        max_index = area_all.index(Max_area)
        A1, B1, C1 = VC[GCI[max_index][0]], VC[GCI[max_index][1]], VC[GCI[max_index][2]]

    return Max_area, A1, B1, C1


def find_coordinate_axes(A, B, C):
    # 计算三条边的长度
    AB = np.linalg.norm(B - A)
    BC = np.linalg.norm(C - B)
    AC = np.linalg.norm(C - A)

    # 找到最长边
    vmax1, vmax2, vmax3 = A, B, C
    if AB < BC:
        vmax1, vmax2, vmax3 = B, C, A
    elif AC > BC:
        vmax1, vmax2, vmax3 = A, C, B

    # 计算最长边的中点
    vo = (vmax1 + vmax2) / 2

    # 计算新坐标系的基向量
    x_axis = (vmax3 - vo) / np.linalg.norm(vmax3 - vo)
    y_axis = (vo - vmax1) / np.linalg.norm(vo - vmax1)
    z_axis = np.cross(x_axis, y_axis)

    return vo, x_axis, y_axis, z_axis


# 原始坐标转换到新的坐标系
def transform_to_new_coordinates(vertices, x_axis, y_axis, z_axis):
    # Create the transformation matrix using the new coordinate axes
    transformation_matrix = np.column_stack((x_axis, y_axis, z_axis))

    # Transform the original vertices to the new coordinate system
    transformed_vertices = np.dot(vertices, transformation_matrix)

    # transformed_vertices_str = [[f'{element:.15f}' for element in row] for row in transformed_vertices]

    return transformed_vertices


# 新坐标系数据还原为原始数据
def revert_to_original_coordinates(transformed_vertices, x_axis, y_axis, z_axis):
    # Create the transformation matrix using the original coordinate axes
    transformation_matrix = np.column_stack((x_axis, y_axis, z_axis))

    # Compute the inverse of the transformation matrix
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

    # Revert the transformed vertices back to the original coordinate system
    original_vertices = np.dot(transformed_vertices, inverse_transformation_matrix)

    return original_vertices


# paillier加密列表
def encrypt_list(lst):
    encrypted_lst = []
    for num in lst:
        encrypted_element = paillier.encrypt(paillier.pk, num)
        encrypted_lst.append(encrypted_element)
    return encrypted_lst


# paillier解密列表
def decrypt_list(encrypted_lst):
    # zmin = dmin
    # zmax = dmax
    decrypted_lst = []
    for row in encrypted_lst:
        if isinstance(row, np.float64):
            #if row == zmin  or row == zmax:
            decrypted_lst.append(row)
            continue
        decrypted_element = paillier.decrypt(paillier.sk, paillier.pk, row)
        decrypted_lst.append(decrypted_element)
    return decrypted_lst


# 新坐标系数据转换为球面坐标
def convert_to_spherical_coordinates(coordinates):
    r = np.linalg.norm(coordinates, axis=1)
    theta = np.arccos(coordinates[:, 2] / r)  # arccos(z/r)
    phi = np.arctan2(coordinates[:, 1], coordinates[:, 0])  # arctan2(y/x)
    return np.column_stack((r, theta, phi))


# 球面坐标转换为笛卡尔坐标系
def convert_to_cartesian_coordinates(spherical_coordinates):
    r = spherical_coordinates[:, 0]
    theta = spherical_coordinates[:, 1]
    phi = spherical_coordinates[:, 2]

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.column_stack((x, y, z))


def convert_to_original_coordinates_from_spherical(spherical_coordinates, x_axis, y_axis, z_axis):
    cartesian_coordinates = convert_to_cartesian_coordinates(spherical_coordinates)

    # Create the transformation matrix using the original coordinate axes
    transformation_matrix = np.column_stack((x_axis, y_axis, z_axis))

    # Compute the inverse of the transformation matrix
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

    # Revert the Cartesian coordinates back to the original coordinate system
    original_coordinates = np.dot(cartesian_coordinates, inverse_transformation_matrix)

    return original_coordinates


def split_columns(input_array):
    # 初始化三个空列表，用于存储每一列的元素
    r = []
    theta = []
    phi = []

    # 遍历二维数组的每一行
    for row in input_array:
        # 将每一列的元素分别添加到对应的列表中
        r.append(row[0])
        theta.append(row[1])
        phi.append(row[2])

    return r, theta, phi


# 计算投影距离
def projection_distance(r_list, theta_list):
    projection_distances = []
    for r, theta in zip(r_list, theta_list):
        distance = r * math.sin(theta)
        projection_distances.append(distance)
    return projection_distances


# 计算Z值距离
def CalculateZ(r_list, theta_list):
    Zsc = []
    for r, theta in zip(r_list, theta_list):
        zsc = r * math.cos(theta)
        Zsc.append(zsc)
    return Zsc


# 数据归一化处理
def min_max_normalize(data):
    R = 1e9  # 1e6
    data_min = min(data)
    data_max = max(data)
    normalized_data = [((x - data_min) / (data_max - data_min)) * R for x in data]
    intnormalized_data = [int(round(x)) for x in normalized_data]
    return intnormalized_data


# 反归一化
def min_max_denormalize(normalized_data, data_min, data_max):
    R = 1e9  # 1e6
    denormalized_data = [x / R * (data_max - data_min) + data_min for x in normalized_data]
    return denormalized_data


# 密文域水印嵌入
def Embed_Watermark(FS: object, normalized_distance: object, intz: object, scz: object) -> object:
    '''Z坐标水印嵌入'''
    encryptedz = []
    matrix_Z = normalized_distance

    scz_min = min(scz)
    scz_max = max(scz)

    for i, row in zip(intz, matrix_Z):
        if i == 0 or i == 1e9:
            i = i / 1e9 * (scz_max - scz_min) + scz_min
            encryptedz.append(i)
            continue
        # 构建索引位，得到一个0 1 序列
        index_x = row % len(FS)
        # 水印为 0 时，使得 lsb 与 slsb 奇偶不同
        if FS[index_x] == 0:
            thousands_digits = (i // 1000) % 10
            hundreds_digits = (i // 100) % 10
            if hundreds_digits % 2 == thousands_digits % 2:
                eni = paillier.encrypt(paillier.pk, i) * paillier.encrypt(paillier.pk, 1000)
                encryptedz.append(eni)
            else:
                eni = paillier.encrypt(paillier.pk, i)
                encryptedz.append(eni)
        # 水印为 1 时，使得 lsb 与 slsb 奇偶相同
        if FS[index_x] == 1:
            hundreds_digits = (i // 100) % 10
            thousands_digits = (i // 1000) % 10
            if hundreds_digits % 2 != thousands_digits % 2:
                eni = paillier.encrypt(paillier.pk, i) * paillier.encrypt(paillier.pk, 1000)
                encryptedz.append(eni)
            else:
                eni = paillier.encrypt(paillier.pk, i)
                encryptedz.append(eni)
    return encryptedz

# 分块数据合并
def Merge_Chunks(Block_Index_Bit, P_normalized_scz, U_normalized_scz):
    result = []
    index_B = 0
    index_C = 0

    for element in Block_Index_Bit:
        if element == 0:
            result.append(P_normalized_scz[index_B])
            index_B += 1
        elif element == 1:
            result.append(U_normalized_scz[index_C])
            index_C += 1
    return result

if __name__ == '__main__':
    # 数据属性信息
    '''
    Number_meshes(NM)：三角网格个数 
    Vertex_coordinates(VC): 顶点坐标二维列表，[[x,y,z],[x,y,z]]
    Grid_coordinate_index(GCI): 三角网格坐标点索引
    Texture_coordinates(TC): 纹理坐标数组
    Textured_imagery(TI): 纹理影像
    '''
    NM, VC, GCI, TC, TI = Get_OBJ(r"D:\Specialized Software\PyCharm\WorkSpace\3DWaterMark\data\4.obj",
                                  r"D:\Specialized Software\PyCharm\WorkSpace\3DWaterMark\data\Tile_+008_+006_0.jpg") #1234: Tile_+008_+006_0 5: Model_0
    # 三维模型数据00001 = {float} 24.733097076416016
    vertices = VC.tolist()

    # 最大三角网格及三个顶点坐标
    Max_Area, A11, B11, C11 = tri_area(VC, GCI)

    # 计算新的坐标轴
    vo, x_axis, y_axis, z_axis = find_coordinate_axes(A11, B11, C11)

    # 将新坐标系原点移到[0，0，0]
    result_list = [[x - y for x, y in zip(sublist, vo)] for sublist in vertices]

    # 将原始坐标转换到新的坐标系下
    transformed_vertices = transform_to_new_coordinates(result_list, x_axis, y_axis, z_axis)
    # reverted_vertices_list = revert_to_original_coordinates(transformed_vertices, x_axis, y_axis, z_axis)
    # reverted_vertices = [[x + y for x, y in zip(sublists, vo)] for sublists in reverted_vertices_list]

    # 计算所有点到给原点的距离的投影长度
    distances = [np.linalg.norm([np.array(point[0]), np.array(point[1]), 0]) for point in transformed_vertices]

    # 进行最大最小归一化
    # data_min = min(distance)
    # data_max = max(distance)
    normalized_distance = min_max_normalize(distances)
    # denormalized_data = min_max_denormalize(normalized_distance, data_min, data_max)
    # print(distance == denormalized_data)

    # Z坐标提取
    scz = [row[2] for row in transformed_vertices]

    # 归一化
    scz_min = min(scz)
    scz_max = max(scz)
    normalized_scz = min_max_normalize(scz)

    # 数据分块
    Block_Index = [int(str(num)[0:4]) for num in normalized_scz]  # [12,23,45,56]
    Block_Index_Bit = [num % 2 for num in Block_Index]  # [0,1,1,0]
    # 用于嵌入发行商追踪指纹
    P_normalized_scz = [normalized_scz[i] for i in range(len(normalized_scz)) if Block_Index_Bit[i] == 0]
    P_normalized_distance = [normalized_distance[i] for i in range(len(normalized_distance)) if Block_Index_Bit[i] == 0]
    # 用于嵌入用户指纹
    U_normalized_scz = [normalized_scz[i] for i in range(len(normalized_scz)) if Block_Index_Bit[i] == 1]
    U_normalized_distance = [normalized_distance[i] for i in range(len(normalized_distance)) if Block_Index_Bit[i] == 1]

    # # PFS:Publisher fingerprint sequence
    PFS = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    # UFS:User fingerprint sequence
    UFS = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]

    # 含发行商指纹密文Z坐标
    P_EncryptedZ = Embed_Watermark(PFS, P_normalized_distance, P_normalized_scz, scz)
    # 含用户指纹密文Z坐标
    U_EncryptedZ = Embed_Watermark(UFS, U_normalized_distance, U_normalized_scz, scz)
    # 已嵌入指纹密文Z坐标
    EncryptedZ = Merge_Chunks(Block_Index_Bit, P_EncryptedZ, U_EncryptedZ)

    DecryptedZ = decrypt_list(EncryptedZ)
    denormalized_scz = min_max_denormalize(DecryptedZ, scz_min, scz_max)
    # denormalized_scz = min_max_denormalize(EncryptedZ, scz_min, scz_max)

    # 含指纹明文域矢量三维模型
    new_list = []
    for i in range(len(transformed_vertices)):
        temp = transformed_vertices[i]
        temp[2] = denormalized_scz[i]
        new_list.append(temp)
    # 坐标系还原与写入
    reverted_vertices_list = revert_to_original_coordinates(new_list, x_axis, y_axis, z_axis)
    reverted_vertices = [[x + y for x, y in zip(sublists, vo)] for sublists in reverted_vertices_list]
    create_newobj(reverted_vertices,
                  r"D:\Specialized Software\PyCharm\WorkSpace\3DWaterMark\data\Tile_+008_+006_0.jpg")

    print("ok")


    ### 单一指纹嵌入
    # # FS:fingerprint sequence
    # FS = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
    #       0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1]
    #
    # # 含指纹密文Z坐标
    # EncryptedZ = Embed_Watermark(FS, normalized_distance, normalized_scz, scz)
    # # ZEncryptedZ = [row / 1e35 for row in EncryptedZ]
    # # NZEncryptedZ = [(rov * 1e35) for rov in ZEncryptedZ]
    # DecryptedZ = decrypt_list(EncryptedZ)
    # denormalized_scz = min_max_denormalize(DecryptedZ, scz_min, scz_max)
    #
    # # 含指纹明文域矢量三维模型
    # new_list = []
    # for i in range(len(transformed_vertices)):
    #     temp = transformed_vertices[i]
    #     temp[2] = denormalized_scz[i]
    #     new_list.append(temp)
    # # 坐标系还原与写入
    # reverted_vertices_list = revert_to_original_coordinates(new_list, x_axis, y_axis, z_axis)
    # reverted_vertices = [[x + y for x, y in zip(sublist, vo)] for sublist in reverted_vertices_list]
    # create_newobj(reverted_vertices,
    #               r"D:\Specialized Software\PyCharm\WorkSpace\3DWaterMark\data\Model_0.jpg")
    #
    # print("ok")


    ### 密文模型生成
    # # 含发行商指纹密文Z坐标
    # P_EncryptedZ = Embed_Watermark(PFS, P_normalized_distance, P_normalized_scz, scz)
    # # 含用户指纹密文Z坐标
    # U_EncryptedZ = Embed_Watermark(UFS, U_normalized_distance, U_normalized_scz, scz)
    # # 已嵌入指纹密文Z坐标
    # EncryptedZ = Merge_Chunks(Block_Index_Bit, P_EncryptedZ, U_EncryptedZ)
    #
    # zoom_EncryptedZ = [x * 10 ** (-65) for x in EncryptedZ]
    #
    # # DecryptedZ = decrypt_list(EncryptedZ, scz_min, scz_max)
    # # denormalized_scz = min_max_denormalize(DecryptedZ, scz_min, scz_max)
    # # denormalized_scz = min_max_denormalize(EncryptedZ, scz_min, scz_max)
    #
    # # 含指纹明文域矢量三维模型
    # new_list = []
    # for i in range(len(transformed_vertices)):
    #     temp = transformed_vertices[i]
    #     temp[2] = zoom_EncryptedZ[i]
    #     new_list.append(temp)
    # # 坐标系还原与写入
    # reverted_vertices_list = revert_to_original_coordinates(new_list, x_axis, y_axis, z_axis)
    # reverted_vertices = [[x + y for x, y in zip(sublist, vo)] for sublist in reverted_vertices_list]
    # create_newobj(reverted_vertices,
    #               r"D:\Specialized Software\PyCharm\WorkSpace\3DWaterMark\data\Tile_+008_+006_0.jpg")
    #
    # print("ok")
