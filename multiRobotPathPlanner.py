#!/usr/bin/env python3
import sys
import time
import numpy as np
from PIL import Image
from darp import DARP
from kruskal import Kruskal
from CalculateTrajectories import CalculateTrajectories
from Visualization import visualize_paths
from turns import turns


def get_area_map(path, area=0, obs=-1):
    le_map = np.array(Image.open(path))
    ma = np.array(le_map).mean(axis=2) != 0
    le_map = np.int8(np.zeros(ma.shape))
    le_map[ma] = area
    le_map[~ma] = obs
    return le_map


def get_area_indices(area, value, inv=False, obstacle=-1):
    try:
        value = int(value)
        if inv:
            return np.concatenate([np.where((area != value))]).T
        return np.concatenate([np.where((area == value))]).T
    except:
        mask = area == value[0]
        if inv:
            mask = area != value[0]
        for v in value[1:]:
            if inv:
                mask &= area != v
            else:
                mask |= area == v
        mask &= area != obstacle
        return np.concatenate([np.where(mask)]).T


class MultiRobotPathPlanner(DARP):
    def __init__(self, nx, ny, notEqualPortions, initial_positions, portions,
                 obs_pos, visualization, MaxIter=80000, CCvariation=0.01,
                 randomLevel=0.0001, dcells=2, importance=False):

        start_time = time.time()
        self.darp_instance = DARP(
            nx, ny, notEqualPortions, initial_positions, portions, obs_pos,
            visualization, MaxIter=MaxIter, CCvariation=CCvariation,
            randomLevel=randomLevel, dcells=dcells, importance=importance
        )

        self.DARP_success, self.iterations = self.darp_instance.divideRegions()

        if not self.DARP_success:
            print("DARP did not manage to find a solution for the given configuration!")
        else:
            self.mode_to_drone_turns = []
            AllRealPaths_dict = {}
            subCellsAssignment_dict = {}
            for mode in range(4):
                MSTs = self.calculateMSTs(
                    self.darp_instance.BinaryRobotRegions,
                    self.darp_instance.droneNo,
                    self.darp_instance.rows,
                    self.darp_instance.cols,
                    mode
                )
                AllRealPaths = []
                for r in range(self.darp_instance.droneNo):
                    ct = CalculateTrajectories(
                        self.darp_instance.rows, self.darp_instance.cols, MSTs[r]
                    )
                    ct.initializeGraph(
                        self.CalcRealBinaryReg(
                            self.darp_instance.BinaryRobotRegions[r],
                            self.darp_instance.rows,
                            self.darp_instance.cols
                        ), True
                    )
                    ct.RemoveTheAppropriateEdges()
                    ct.CalculatePathsSequence(
                        4 * self.darp_instance.initial_positions[r][0] * self.darp_instance.cols +
                        2 * self.darp_instance.initial_positions[r][1]
                    )
                    raw_path = ct.PathSequence

                    simplified = self.reduce_path_density(raw_path)
                    AllRealPaths.append(simplified)

                # (Остальной код визуализации TypesOfLines и прочее можно оставить,
                #  но так как мы теперь не используем эти данные для публикации,
                #  их можно закомментировать для ускорения. Я оставил для совместимости.)
                self.TypesOfLines = np.zeros((self.darp_instance.rows * 2, self.darp_instance.cols * 2, 2))
                for r in range(self.darp_instance.droneNo):
                    flag = False
                    for connection in AllRealPaths[r]:
                        if flag:
                            if self.TypesOfLines[connection[0]][connection[1]][0] == 0:
                                indxadd1 = 0
                            else:
                                indxadd1 = 1
                            if self.TypesOfLines[connection[2]][connection[3]][0] == 0 and flag:
                                indxadd2 = 0
                            else:
                                indxadd2 = 1
                        else:
                            if not (self.TypesOfLines[connection[0]][connection[1]][0] == 0):
                                indxadd1 = 0
                            else:
                                indxadd1 = 1
                            if not (self.TypesOfLines[connection[2]][connection[3]][0] == 0 and flag):
                                indxadd2 = 0
                            else:
                                indxadd2 = 1

                        flag = True
                        if connection[0] == connection[2]:
                            if connection[1] > connection[3]:
                                self.TypesOfLines[connection[0]][connection[1]][indxadd1] = 2
                                self.TypesOfLines[connection[2]][connection[3]][indxadd2] = 3
                            else:
                                self.TypesOfLines[connection[0]][connection[1]][indxadd1] = 3
                                self.TypesOfLines[connection[2]][connection[3]][indxadd2] = 2
                        else:
                            if connection[0] > connection[2]:
                                self.TypesOfLines[connection[0]][connection[1]][indxadd1] = 1
                                self.TypesOfLines[connection[2]][connection[3]][indxadd2] = 4
                            else:
                                self.TypesOfLines[connection[0]][connection[1]][indxadd1] = 4
                                self.TypesOfLines[connection[2]][connection[3]][indxadd2] = 1

                subCellsAssignment = np.zeros((2 * self.darp_instance.rows, 2 * self.darp_instance.cols))
                for i in range(self.darp_instance.rows):
                    for j in range(self.darp_instance.cols):
                        subCellsAssignment[2 * i][2 * j] = self.darp_instance.A[i][j]
                        subCellsAssignment[2 * i + 1][2 * j] = self.darp_instance.A[i][j]
                        subCellsAssignment[2 * i][2 * j + 1] = self.darp_instance.A[i][j]
                        subCellsAssignment[2 * i + 1][2 * j + 1] = self.darp_instance.A[i][j]

                drone_turns = turns(AllRealPaths)
                drone_turns.count_turns()
                drone_turns.find_avg_and_std()
                self.mode_to_drone_turns.append(drone_turns)
                AllRealPaths_dict[mode] = AllRealPaths
                subCellsAssignment_dict[mode] = subCellsAssignment

            averages = [x.avg for x in self.mode_to_drone_turns]
            self.min_mode = averages.index(min(averages))

            combined_modes_paths = []
            combined_modes_turns = []
            for r in range(self.darp_instance.droneNo):
                min_turns = sys.maxsize
                best_path = []
                for mode in range(4):
                    if self.mode_to_drone_turns[mode].turns[r] < min_turns:
                        best_path = self.mode_to_drone_turns[mode].paths[r]
                        min_turns = self.mode_to_drone_turns[mode].turns[r]
                combined_modes_paths.append(best_path)
                combined_modes_turns.append(min_turns)

            self.best_case = turns(combined_modes_paths)
            self.best_case.turns = combined_modes_turns
            self.best_case.find_avg_and_std()

            if self.darp_instance.visualization:
                image = visualize_paths(
                    self.best_case.paths,
                    subCellsAssignment_dict[self.min_mode],
                    self.darp_instance.droneNo,
                    self.darp_instance.color
                )
                image.visualize_paths("Combined Modes")

            self.execution_time = time.time() - start_time
            best_case_num_paths = [len(x) for x in self.best_case.paths]
            print(f'\nResults:')
            print(f'Number of cells per robot: {best_case_num_paths}')
            print(f'Minimum number of cells in robots paths: {min(best_case_num_paths)}')
            print(f'Maximum number of cells in robots paths: {max(best_case_num_paths)}')
            print(f'Average number of cells in robots paths: {np.mean(np.array(best_case_num_paths))}')
            print(f'\nTurns Analysis: {self.best_case}')

    def reduce_path_density(self, connections):
        """
        Удаляет каждую вторую точку на прямых участках пути,
        сохраняя все повороты и замыкание.
        """
        if not connections:
            return connections

        # Преобразуем соединения в список точек
        points = [(connections[0][0], connections[0][1])]
        for conn in connections:
            points.append((conn[2], conn[3]))

        # Удаляем последовательные дубликаты
        pts = [points[0]]
        for p in points[1:]:
            if p != pts[-1]:
                pts.append(p)

        if len(pts) < 2:
            return connections

        new_pts = [pts[0]]
        i = 0
        n = len(pts)
        while i < n - 1:
            r0, c0 = pts[i]
            r1, c1 = pts[i+1]
            # Определяем направление первого шага
            if r0 == r1:      # горизонтальное движение
                j = i + 1
                while j < n and pts[j][0] == r0:
                    j += 1
            else:            # вертикальное движение (c0 == c1)
                j = i + 1
                while j < n and pts[j][1] == c0:
                    j += 1
            # Прямой участок: pts[i:j]
            segment = pts[i:j]
            # Оставляем каждую вторую точку, но всегда сохраняем начало и конец
            kept = [segment[0]]
            for k in range(1, len(segment)-1):
                if k % 2 == 0:   # оставляем чётные индексы (0,2,4,...)
                    kept.append(segment[k])
            kept.append(segment[-1])
            # Фильтруем возможные дубликаты внутри kept
            filtered = [kept[0]]
            for p in kept[1:]:
                if p != filtered[-1]:
                    filtered.append(p)
            # Добавляем все точки кроме первой (она уже есть в new_pts)
            new_pts.extend(filtered[1:])
            i = j - 1   # последняя точка сегмента — начало следующего

        # Финальная очистка от дубликатов
        final_pts = [new_pts[0]]
        for p in new_pts[1:]:
            if p != final_pts[-1]:
                final_pts.append(p)

        if len(final_pts) < 2:
            return connections

        # Обратно в формат соединений
        new_connections = []
        for idx in range(len(final_pts)-1):
            a = final_pts[idx]
            b = final_pts[idx+1]
            new_connections.append((a[0], a[1], b[0], b[1]))
        return new_connections

    def CalcRealBinaryReg(self, BinaryRobotRegion, rows, cols):
        temp = np.zeros((2 * rows, 2 * cols))
        RealBinaryRobotRegion = np.zeros((2 * rows, 2 * cols), dtype=bool)
        for i in range(2 * rows):
            for j in range(2 * cols):
                temp[i, j] = BinaryRobotRegion[(int(i / 2))][(int(j / 2))]
                if temp[i, j] == 0:
                    RealBinaryRobotRegion[i, j] = False
                else:
                    RealBinaryRobotRegion[i, j] = True
        return RealBinaryRobotRegion

    def calculateMSTs(self, BinaryRobotRegions, droneNo, rows, cols, mode):
        MSTs = []
        for r in range(droneNo):
            k = Kruskal(rows, cols)
            k.initializeGraph(BinaryRobotRegions[r, :, :], True, mode)
            k.performKruskal()
            MSTs.append(k.mst)
        return MSTs


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-grid', default=(10, 10), type=int, nargs=2)
    argparser.add_argument('-obs_pos', default=[5, 6, 7], nargs='*', type=int)
    argparser.add_argument('-in_pos', default=[0, 3, 9], nargs='*', type=int)
    argparser.add_argument('-nep', action='store_true')
    argparser.add_argument('-portions', default=[0.2, 0.3, 0.5], nargs='*', type=float)
    argparser.add_argument('-vis', action='store_true')
    args = argparser.parse_args()

    MultiRobotPathPlanner(
        args.grid[0], args.grid[1], args.nep, args.in_pos, args.portions,
        args.obs_pos, args.vis
    )