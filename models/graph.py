import numpy as np

class Graph:
    def __init__(self, layout="coco", strategy="spatial", max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def get_edge(self, layout):
        if layout == "coco":
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (0, 1), (0, 2), (1, 3), (2, 4),
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (5, 11), (6, 12)
            ]
            self.edge = self_link + neighbor_link
            self.center = 0

        elif layout in ["kinect25", "ntu25", "ntu-rgb+d"]:
            # NTU/Kinect v2 25-joint topology (0-indexed)
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (0, 1),
                (1, 20), (2, 20), (2, 3),
                (4, 20), (4, 5), (5, 6), (6, 7),
                (8, 20), (8, 9), (9, 10), (10, 11),
                (0, 12), (12, 13), (13, 14), (14, 15),
                (0, 16), (16, 17), (17, 18), (18, 19),
                (21, 22), (22, 7),
                (23, 24), (24, 11),
            ]
            self.edge = self_link + neighbor_link
            self.center = 20  # spine (joint 21 in 1-indexed)

        else:
            raise ValueError(f"Do not support this layout: {layout}")

    @staticmethod
    def get_hop_distance(num_node, edge, max_hop=1):
        A = np.zeros((num_node, num_node), dtype=np.float32)
        for i, j in edge:
            A[i, j] = 1
            A[j, i] = 1

        hop_dis = np.full((num_node, num_node), np.inf, dtype=np.float32)
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node), dtype=np.float32)
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1

        normalize_adjacency = self.normalize_digraph(adjacency)

        if strategy == "spatial":
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node), dtype=np.float32)
                a_close = np.zeros((self.num_node, self.num_node), dtype=np.float32)
                a_further = np.zeros((self.num_node, self.num_node), dtype=np.float32)

                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]

                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)

            self.A = np.stack(A)
        else:
            raise ValueError(f"Do not support this strategy: {strategy}")

    @staticmethod
    def normalize_digraph(A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node), dtype=np.float32)
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        return A @ Dn
