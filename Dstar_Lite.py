import numpy as np
idfhsd;ghsehgpdsrfjgdspnfgds'png
#Khởi tạo kích thước của 1 ô voxel
VOXEL_SIZE = 0.05 # 5cm

X_RANGE = (-0.5, 0.5)
Y_RANGE = (-0.5, 0.5)
Z_RANGE = (0, 0.7)

GRID_X = int((X_RANGE[1] - X_RANGE[0]) / VOXEL_SIZE)
GRID_Y = int((Y_RANGE[1] - Y_RANGE[0]) / VOXEL_SIZE)
GRID_Z = int((Z_RANGE[1] - Z_RANGE[0]) / VOXEL_SIZE)


class DStarLite:
    def __init__(self, start, goal, k_m):
        self.g = {}
        self.rhs = {}
        for i in range(GRID_X):
            for j in range(GRID_Y):
                for k in range(GRID_Z):
                    self.g[(i, j, k)] = float('inf')
                    self.rhs[(i, j, k)] = float('inf')
        self.queue = {}
        self.goal = goal
        self.start = start
        self.k_m = k_m
        self.rhs[goal] = 0
        self.queue[goal] = self.calculateKey(goal, start, k_m)
        self.parents = {}
        self.children = {}
        self.barriers = set()
        self.Parents_Children()

    def Parents_Children(self):
        directions = [  # 6 hướng: +/-X, +/-Y, +/-Z
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1)
        ]

        for i in range(GRID_X):
            for j in range(GRID_Y):
                for k in range(GRID_Z):
                    node = (i, j, k)
                    self.children[node] = []
                    self.parents[node] = []
                    for dx, dy, dz in directions:
                        ni, nj, nk = i + dx, j + dy, k + dz
                        if 0 <= ni < GRID_X and 0 <= nj < GRID_Y and 0 <= nk < GRID_Z:
                            neighbor = (ni, nj, nk)
                            self.children[node].append(neighbor)
                            self.parents[node].append(neighbor)



    def calculateKey(self, node, current, k_m):
        return [min(self.g[node], self.rhs[node]) + self.h(node, current) + k_m, min(self.g[node], self.rhs[node])]

    def h(self, p1, p2):
        # Manhattan distance in 3D space — phù hợp với 6-connectivity
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        return abs(x2 - x1) + abs(y2 - y1) + abs(z2 - z1)
    def topKey(self):
        if len(self.queue) > 0:
            s = min(self.queue, key = self.queue.get)
        # Lấy node có key nhỏ nhất theo tuple so sánh
            return s, self.queue[s]
        else:
            return None, [float('inf'), float('inf')]
        
    def computeShortestPath(self, start, k_m):
        while (self.rhs[start] != self.g[start]) or (self.topKey()[1] < self.calculateKey(start, start, k_m)):
            u, k_old = self.topKey()
            self.queue.pop(u)
            if k_old < self.calculateKey(u, start, k_m):
                self.queue[u] = self.calculateKey(u, start, k_m)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for i in self.parents[u]:
                    self.updateVertex(i, start, k_m)
            else:
                self.g[u] = float('inf')
                self.updateVertex(u, start, k_m)
                for i in self.parents[u]:
                    self.updateVertex(i, start, k_m)

    def updateVertex(self, id, current, k_m):
        if id != self.goal:
            self.rhs[id] = min(self.g[i]+ self.cost(i,id) for i in self.children[id])
        if id in self.queue:
            self.queue.pop(id)
        if self.rhs[id] != self.g[id]:
            self.queue[id] = self.calculateKey(id, current, k_m)
    
    def cost(self,p1, p2):
        if p1 in self.barriers or p2 in self.barriers:
            return float('inf')
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        return abs(x2-x1) + abs(y2-y1) + abs(z2-z1) 
    def reconstruct_path(self, s_last):
        path = [s_last]
        s = s_last
        while s != self.goal:
            s = min((x for x in self.children[s]), key=lambda x: self.g[x]) #if not x.isbarrier()
            path.append(s)
        return path   

def main():
    start = (0, 0,0)
    goal = (0, 1, 8)
    k_m = 0
    planner = DStarLite(start, goal, k_m)
    planner.computeShortestPath()
    print("Giá trị g(start):", planner.g[start])


if __name__ == "__main__":
    main()

