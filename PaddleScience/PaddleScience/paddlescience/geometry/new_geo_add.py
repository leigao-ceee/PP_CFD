import numpy as np
import math
# from .geometry import Geometry

class NGD():
################# 细化部分节点
    # space_origin=(-0.2, -0.2), 
    # space_extent=(0.6, 0.2))
    # init functionß
    def __init__(self, space_origin=None, space_extent=None):
        self.space_origin = space_origin
        self.space_extent = space_extent
        # self.space_ndims = leo

    def bc_index_calc(self):
        # self.space_origin = space_origin
        # self.space_extent = space_extent
        # region1：中间区域网格，包含圆柱，x方向长度为0.3，离散11个点，y方向宽度为0.2，离散11个点
        space_origin_1 = (-0.1, -0.1)
        space_extent_1 = (0.2, 0.1)
        space_nsteps1 = (101, 101)
        nx1 = space_nsteps1[0]  # x方向离散有301个点
        ny1 = space_nsteps1[1]  # y方向离散有201个点


        # region2：右侧区域网格，x方向长度为0.4，离散11个点，y方向宽度为0.4，离散11个点
        space_origin_2 = (0.2, self.space_origin[1])
        space_extent_2 = self.space_extent
        space_nsteps2 = (101, 81)
        nx2 = space_nsteps2[0]  # x方向离散有101个点
        ny2 = space_nsteps2[1]  # y方向离散有101个点

        # region3：上侧网格网格粗，x方向长度为0.3，离散11个点，y方向宽度为0.1，离散5个点
        space_origin_3 = (-0.1, 0.1)
        space_extent_3 = (0.2, self.space_extent[1])
        space_nsteps3 = (81, 41)
        nx3 = space_nsteps3[0]  # x方向离散有11个点
        ny3 = space_nsteps3[1]  # y方向离散有11个点

        # region4：下侧网格网格粗，x方向长度为0.3，离散11个点，y方向宽度为0.1，离散5个点
        space_origin_4 = (-0.1, self.space_origin[1])
        space_extent_4 = (0.2, -0.1)
        space_nsteps4 = (81, 41)
        nx4 = space_nsteps4[0]  # x方向离散有11个点
        ny4 = space_nsteps4[1]  # y方向离散有11个点

        # region5：左侧区域，网格粗，x方向长度为0.1，离散11个点，y方向宽度为0.4，离散11个点
        space_origin_5 = self.space_origin
        space_extent_5 = (-0.1, self.space_extent[1])
        space_nsteps5 = (41, 81)
        nx5 = space_nsteps5[0]  # x方向离散有11个点
        ny5 = space_nsteps5[1]  # y方向离散有11个点

        # 定义区域1的节点
        steps1 = []  # steps是在二维的平面内，将x方向从-0.05到0.05进行均匀离散，并记录所有的x方向的坐标，同时对于y方向也同样，最后得到包含两个数组的坐标集
        for i in range(self.space_ndims):
            steps1.append(
                np.linspace(
                    space_origin_1[i],
                    space_extent_1[i],
                    space_nsteps1[i],
                    endpoint=True))

        # 对x方向给定的向量与y方向给定的向量进行网格化，得到内部坐标（uniform形式）,mesh中两个矩阵，分别代表x，y
        mesh1 = np.meshgrid(steps1[1], steps1[0], sparse=False, indexing='ij')
        # print(mesh1)
        # 构建网格区域，将mesh中的两个矩阵展平，并整合成一组完整的坐标(x,y)，对应的是domain[index][0]表示网格点的x位置，domain[index][1]表示网格点的y位置。
        domain1 = np.stack(
            (mesh1[1].reshape(-1), mesh1[0].reshape(-1)), axis=-1)
        space_domain1 = domain1
        # print(space_domain1.shape)

        theta = 0
        x_circle = []
        y_circle = []
        theta = 0
        r = 0.05
        while theta <= 2 * math.pi:
            x_circle = np.append(x_circle, r * math.cos(theta))
            y_circle = np.append(y_circle, r * math.sin(theta))
            theta += 0.0005
        x_c1 = x_circle.reshape(len(x_circle), 1)
        y_c1 = y_circle.reshape(len(y_circle), 1)
        xy_circle = np.hstack([x_c1, y_c1])

        # 删除domain中圆柱内部的坐标点
        index_o = 0
        index_i = 0
        bc_inner = []
        for point in space_domain1:
            for inner_p in xy_circle:
                if abs(inner_p[0]) - abs(point[0]) > 1e-5 and abs(inner_p[1]) - abs(point[1]) > 1e-5:
                    index_i = index_o
                    bc_inner.append(index_i)
            index_o += 1
        space_domain1 = np.delete(space_domain1, np.unique(bc_inner), axis=0)
        # print(space_domain1.shape)

        # 定义区域2的节点
        steps2 = []  # steps是在二维的平面内，将x方向从-0.05到0.05进行均匀离散，并记录所有的x方向的坐标，同时对于y方向也同样，最后得到包含两个数组的坐标集
        for i in range(self.space_ndims):
            steps2.append(
                np.linspace(
                    space_origin_2[i],
                    space_extent_2[i],
                    space_nsteps2[i],
                    endpoint=True))

        # 对x方向给定的向量与y方向给定的向量进行网格化，得到内部坐标（uniform形式）,mesh中两个矩阵，分别代表x，y
        mesh2 = np.meshgrid(steps2[1], steps2[0], sparse=False, indexing='ij')
        # print(mesh2)
        # 构建网格区域，将mesh中的两个矩阵展平，并整合成一组完整的坐标(x,y)，对应的是domain[index][0]表示网格点的x位置，domain[index][1]表示网格点的y位置。
        domain2 = np.stack(
            (mesh2[1].reshape(-1), mesh2[0].reshape(-1)), axis=-1)
        space_domain2 = domain2
        # print(space_domain2.shape)

        # 定义区域3的节点
        steps3 = []  # steps是在二维的平面内，将x方向从-0.05到0.05进行均匀离散，并记录所有的x方向的坐标，同时对于y方向也同样，最后得到包含两个数组的坐标集
        for i in range(self.space_ndims):
            steps3.append(
                np.linspace(
                    space_origin_3[i],
                    space_extent_3[i],
                    space_nsteps3[i],
                    endpoint=True))

        # 对x方向给定的向量与y方向给定的向量进行网格化，得到内部坐标（uniform形式）,mesh中两个矩阵，分别代表x，y
        mesh3 = np.meshgrid(steps3[1], steps3[0], sparse=False, indexing='ij')
        # print(mesh2)
        # 构建网格区域，将mesh中的两个矩阵展平，并整合成一组完整的坐标(x,y)，对应的是domain[index][0]表示网格点的x位置，domain[index][1]表示网格点的y位置。
        domain3 = np.stack(
            (mesh3[1].reshape(-1), mesh3[0].reshape(-1)), axis=-1)
        space_domain3 = domain3
        # print(space_domain3.shape)

        # 定义区域4的节点
        steps4 = []  # steps是在二维的平面内，将x方向从-0.05到0.05进行均匀离散，并记录所有的x方向的坐标，同时对于y方向也同样，最后得到包含两个数组的坐标集
        for i in range(self.space_ndims):
            steps4.append(
                np.linspace(
                    space_origin_4[i],
                    space_extent_4[i],
                    space_nsteps4[i],
                    endpoint=True))

        # 对x方向给定的向量与y方向给定的向量进行网格化，得到内部坐标（uniform形式）,mesh中两个矩阵，分别代表x，y
        mesh4 = np.meshgrid(steps4[1], steps4[0], sparse=False, indexing='ij')
        # print(mesh2)
        # 构建网格区域，将mesh中的两个矩阵展平，并整合成一组完整的坐标(x,y)，对应的是domain[index][0]表示网格点的x位置，domain[index][1]表示网格点的y位置。
        domain4 = np.stack(
            (mesh4[1].reshape(-1), mesh4[0].reshape(-1)), axis=-1)
        space_domain4 = domain4
        # print(space_domain4.shape)

        # 定义区域5的节点
        steps5 = []  # steps是在二维的平面内，将x方向从-0.05到0.05进行均匀离散，并记录所有的x方向的坐标，同时对于y方向也同样，最后得到包含两个数组的坐标集
        for i in range(self.space_ndims):
            steps5.append(
                np.linspace(
                    space_origin_5[i],
                    space_extent_5[i],
                    space_nsteps5[i],
                    endpoint=True))

        # 对x方向给定的向量与y方向给定的向量进行网格化，得到内部坐标（uniform形式）,mesh中两个矩阵，分别代表x，y
        mesh5 = np.meshgrid(steps5[1], steps5[0], sparse=False, indexing='ij')
        # print(mesh2)
        # 构建网格区域，将mesh中的两个矩阵展平，并整合成一组完整的坐标(x,y)，对应的是domain[index][0]表示网格点的x位置，domain[index][1]表示网格点的y位置。
        domain5 = np.stack(
            (mesh5[1].reshape(-1), mesh5[0].reshape(-1)), axis=-1)
        space_domain5 = domain5
        # print(space_domain5.shape)

        space_domain = np.vstack((space_domain1, space_domain2, space_domain3, space_domain4, space_domain5))
        # sys_domain = np.unique(sys_domain)
        # print(space_domain.shape)

        space_domain = np.array(list(set([tuple(t) for t in space_domain])))
        # print(f'space_domian.shape:{space_domain.shape}')

        # plt.scatter(space_domain[:, 0], space_domain[:, 1])
        # plt.show()

        # 定义完整bc_index:
        # 1) 定义圆柱周边边界
        nbc = 0
        id_o = 0
        index_c = 0
        bc_cir = []
        for pp in space_domain:
            for bp in xy_circle:
                if abs(bp[0] - pp[0]) <= 0.0001 and abs(bp[1] - pp[1]) <= 0.0001:
                    index_c = id_o
                    bc_cir.append(index_c)
            id_o += 1
        bc_cir = np.unique(bc_cir)
        nbc += len(bc_cir)  # 记录当前bc_index中包含的元素数量,代表圆柱边界的点数量

        # print("test"*3)

        # 2) 定义外围边界
        id_r = 0
        index_b = 0
        bc_side = []
        for pp in space_domain:
            if abs(space_origin_5[0] - pp[0]) < 0.0001 or abs(space_extent_2[0] - pp[0]) < 0.0001 or abs(
                    space_origin_5[1] - pp[1]) < 0.0001 or abs(space_extent_2[1] - pp[1]) < 0.0001:
                index_b = id_r
                bc_side.append(index_b)
            id_r += 1
        bc_side = np.unique(bc_side)

        # print("tttt"*4)
        # 定义边界点总数
        nbc += len(bc_side)

        # 定义bc_index,指的是domain中点的id数组
        base_bc_index = np.vstack((bc_side.reshape(-1, 1), bc_cir.reshape(-1, 1)))

        bc_index = np.ndarray(nbc, dtype=int)
        for i in range(len(base_bc_index)):
            bc_index[i] = base_bc_index[i][0]
        return bc_index, space_domain
