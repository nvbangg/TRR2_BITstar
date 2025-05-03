import pygame
import random
import heapq
import math
import time
from collections import defaultdict

# Màu sắc
WHITE, RED, BLUE, GREEN, BLACK = (
    (255, 255, 255),
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
    (0, 0, 0),
)

# Hằng số
INF = float("inf")
FPS = 60
WIDTH, HEIGHT = SIZE_MAP = (1200, 600)
SO_VAT = 20
SIZE_VAT = (100, 100)
START = (200, HEIGHT - 200)
GOAL = (WIDTH - 200, 200)
MAX_SO_LAN_LAP = 10
SIZE_LO = 25
R_NUT = 3
DO_RONG_CANH = 2


class Node:
    def __init__(self, state):
        self.state = state
        self.g = self._khoang_cach(state, START)
        self.h = self._khoang_cach(state, GOAL)
        self.f = self.g + self.h
        self.g_T = INF
        self.parent = self

    def __lt__(self, other):
        self_cost = self.cost()
        other_cost = other.cost()
        if self_cost == other_cost:
            return self.g_T < other.g_T
        return self_cost < other_cost

    def cost(self):
        return min(self.g_T + self.h, INF)

    @staticmethod
    def _khoang_cach(pos1, pos2):
        return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])


class BITStar:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SIZE_MAP)
        pygame.display.set_caption("BIT* Path Planning")
        self.font = pygame.font.SysFont("Arial", 24)
        self.clock = pygame.time.Clock()

        self.V = []
        self.E = []
        self.X_unconn = []
        self.vat_vat = []
        self.edge_dict = defaultdict(list)

        self.nut_start = Node(START)
        self.nut_start.g_T = 0
        self.nut_end = Node(GOAL)
        self.V.append(self.nut_start)

        self.r = INF
        self.Q_E = []
        self.Q_V = []
        self.V_old = []
        self.path_new = []
        self.path_old = []
        self.c_i = INF
        self.running = True
        self.so_lan_lap = 0
        self.finished = False

    def trong_vat(self, state):
        return not any(obs.collidepoint(state) for obs in self.vat_vat)

    def trong_elip(self, state):
        if self.c_i == INF:
            return True
        dist_start = math.hypot(state[0] - START[0], state[1] - START[1])
        dist_goal = math.hypot(state[0] - GOAL[0], state[1] - GOAL[1])
        return dist_start + dist_goal <= self.c_i

    def co_va_cham(self, v, w):
        x1, y1 = v.state
        x2, y2 = w.state
        dist = Node._khoang_cach((x1, y1), (x2, y2))
        steps = min(int(dist / 5) + 5, 50)

        for i in range(steps):
            u = i / (steps - 1)
            pos = (u * x2 + (1 - u) * x1, u * y2 + (1 - u) * y1)
            if not self.trong_vat(pos):
                return True
        return False

    def khoang_cach(self, v, w):
        return Node._khoang_cach(v.state, w.state)

    def chi_phi_canh(self, v, w):
        return INF if self.co_va_cham(v, w) else self.khoang_cach(v, w)

    def cap_nhat_r(self):
        n = max(1, len(self.V) + len(self.X_unconn))
        return (
            2
            * 1.1
            * ((1 + 1 / 2) * (WIDTH * HEIGHT / math.pi) ** 0.5)
            * ((math.log(n) / n) ** 0.5)
        )

    def gia_tri_tot_nhat(self, queue):
        if not queue:
            return INF
        if isinstance(queue[0], tuple):
            return queue[0][0]
        else:
            return queue[0].cost()

    def tao_vat(self):
        for _ in range(SO_VAT):
            while True:
                pos = (
                    random.randint(0, WIDTH - SIZE_VAT[0]),
                    random.randint(0, HEIGHT - SIZE_VAT[1]),
                )
                obs = pygame.Rect(pos, SIZE_VAT)
                if not (obs.collidepoint(START) or obs.collidepoint(GOAL)):
                    self.vat_vat.append(obs)
                    break

    def cat_tia(self):
        # Cắt đỉnh và mẫu nằm ngoài elip
        self.X_unconn = [x for x in self.X_unconn if x.f < self.c_i]
        self.V = [v for v in self.V if v.f <= self.c_i]

        # Lưu cạnh cũ và khởi tạo lại danh sách cạnh
        old_E = self.E
        self.E = []
        self.edge_dict.clear()

        # Xử lý các cạnh cũ
        for v, w in old_E:
            if v.f <= self.c_i and w.f <= self.c_i:
                self.E.append((v, w))
                self.edge_dict[v].append(w)
            else:
                w.g_T = INF
                w.parent = w

        to_move = [v for v in self.V if v.g_T == INF]
        self.X_unconn.extend(to_move)
        self.V = [v for v in self.V if v.g_T < INF]
        self.update_display(wait=True)

    def them_canh(self, v, w):
        if v.g_T < INF:
            w.g_T = v.g_T + self.khoang_cach(v, w)
            w.parent = v
            self.E.append((v, w))
            self.edge_dict[v].append(w)

    def hang_doi_canh(self, v, x):
        if v.g_T < INF:
            cost_val = v.g_T + self.khoang_cach(v, x) + x.h
            heapq.heappush(self.Q_E, (cost_val, (v, x)))

    def hang_doi_dinh(self, v):
        heapq.heappush(self.Q_V, (v.cost(), v))

    def mo_rong_dinh(self):
        _, v = heapq.heappop(self.Q_V)
        if v.g_T == INF:
            return

        kc_limit = self.r
        g_T_v = v.g_T
        g_T_dest = self.nut_end.g_T

        # Xử lý đỉnh chưa kết nối
        for x in self.X_unconn:
            kc = self.khoang_cach(v, x)
            if (
                kc <= kc_limit
                and g_T_v + kc + x.h < g_T_dest
                and not self.co_va_cham(v, x)
            ):
                self.hang_doi_canh(v, x)

        # Xử lý đỉnh đã có trong V
        if v not in self.V_old:
            for w in self.V:
                if v != w and w not in self.edge_dict.get(v, []):
                    kc = self.khoang_cach(v, w)
                    if (
                        kc <= kc_limit
                        and g_T_v + kc + w.h < g_T_dest
                        and g_T_v + kc < w.g_T
                        and not self.co_va_cham(v, w)
                    ):
                        self.hang_doi_canh(v, w)

    def duong_den_dich(self):
        if self.nut_end.g_T == INF:
            return []
        path, v = [], self.nut_end
        while v != self.nut_start:
            path.append(v)
            v = v.parent
        path.append(self.nut_start)
        return path

    def xu_ly_lo(self):
        # Khởi tạo
        if self.so_lan_lap == 0:
            self.V, self.X_unconn = [self.nut_start], [self.nut_end]
        else:
            self.cat_tia()

        self.Q_E, self.Q_V = [], []
        j = 0

        while True:
            # Kiểm tra điều kiện dừng
            if not self.Q_E and not self.Q_V:
                j += 1
                if (
                    j > 5
                    or self.so_lan_lap >= MAX_SO_LAN_LAP
                    or self.nut_end.g_T < self.c_i
                ):
                    if self.nut_end.g_T < self.c_i:
                        self.so_lan_lap += 1
                    else:
                        self.running = False
                    break

                # Lấy mẫu mới
                new_samples = self.lay_mau()
                if not new_samples:
                    print("Không tạo được mẫu mới.")
                    self.running = False
                    break

                self.X_unconn.extend(new_samples)
                self.cat_tia()
                self.V_old = self.V.copy()
                for v in self.V:
                    self.hang_doi_dinh(v)
                self.r = self.cap_nhat_r()
                continue

            # Mở rộng đỉnh tốt nhất
            best_edge = self.gia_tri_tot_nhat(self.Q_E)
            while self.Q_V and self.gia_tri_tot_nhat(self.Q_V) <= best_edge:
                self.mo_rong_dinh()
                best_edge = self.gia_tri_tot_nhat(self.Q_E)

            if not self.Q_E:
                continue

            # Xử lý cạnh tốt nhất
            _, (v_m, x_m) = heapq.heappop(self.Q_E)
            g_T_vm = v_m.g_T
            kc = self.khoang_cach(v_m, x_m)

            if g_T_vm + kc + x_m.h >= self.nut_end.g_T or g_T_vm + kc >= x_m.g_T:
                continue

            if not self.co_va_cham(v_m, x_m) and g_T_vm + kc + x_m.h < self.nut_end.g_T:
                # Cập nhật đồ thị
                if x_m in self.V:
                    self.E = [(v, w) for v, w in self.E if w != x_m]
                    for v in list(self.edge_dict.keys()):
                        if x_m in self.edge_dict[v]:
                            self.edge_dict[v].remove(x_m)
                    x_m.parent, x_m.g_T = x_m, INF
                else:
                    self.X_unconn.remove(x_m)
                    self.V.append(x_m)
                    self.hang_doi_dinh(x_m)
                self.them_canh(v_m, x_m)

                x_m_g_T = x_m.g_T
                self.Q_E = [
                    (c, (v, x))
                    for c, (v, x) in self.Q_E
                    if x != x_m or v.g_T + self.khoang_cach(v, x) < x_m_g_T
                ]
                heapq.heapify(self.Q_E)

                if not self.Q_E or not self.Q_V:
                    self.update_display(wait=True)
            else:
                self.Q_E, self.Q_V = [], []

        # Cập nhật đường đi
        if self.c_i < INF:
            self.path_old = self.path_new.copy()
        new_path = self.duong_den_dich()
        if new_path:
            self.path_new, self.c_i = new_path, self.nut_end.g_T
        self.update_display(wait=True)

    def update_display(self, wait=False):
        self.screen.fill(WHITE)

        # Vẽ các vật và điểm đầu cuối
        pygame.draw.circle(self.screen, RED, START, R_NUT + 5)
        pygame.draw.circle(self.screen, RED, GOAL, R_NUT + 5)
        for obs in self.vat_vat:
            pygame.draw.rect(self.screen, BLACK, obs)

        # Vẽ elip
        if self.c_i != INF:
            sx, sy, gx, gy = START[0], START[1], GOAL[0], GOAL[1]
            cx, cy = (sx + gx) / 2, (sy + gy) / 2
            dx, dy = gx - sx, gy - sy
            dist = math.hypot(dx, dy)
            if dist > 0:
                angle = math.atan2(dy, dx)
                a = self.c_i / 2
                b = math.sqrt(max(0, a**2 - (dist / 2) ** 2))

                surf = pygame.Surface((2 * a, 2 * b), pygame.SRCALPHA)
                pygame.draw.ellipse(surf, BLACK, surf.get_rect(center=(a, b)), width=2)
                rot_surf = pygame.transform.rotate(surf, -math.degrees(angle))
                self.screen.blit(rot_surf, rot_surf.get_rect(center=(cx, cy)))

        # Vẽ đồ thị
        for v in self.V:
            if v.g_T < INF:
                pygame.draw.circle(self.screen, BLUE, v.state, R_NUT)

        for v, w in self.E:
            if v.g_T < INF and w.g_T < INF:
                pygame.draw.line(self.screen, BLUE, v.state, w.state, DO_RONG_CANH)

        for x in self.X_unconn:
            pygame.draw.circle(self.screen, GREEN, x.state, R_NUT)

        # Vẽ đường đi tốt nhất
        if self.path_new:
            prev = self.path_new[0]
            for v in self.path_new[1:]:
                pygame.draw.line(
                    self.screen, RED, prev.state, v.state, DO_RONG_CANH + 2
                )
                pygame.draw.circle(self.screen, RED, v.state, R_NUT + 2)
                prev = v

        # Hiển thị chi phí
        cost_text = (
            f"Best cost: {self.c_i:.2f}" if self.c_i != INF else "Best cost: INF"
        )
        self.screen.blit(self.font.render(cost_text, True, BLACK), (10, 10))

        pygame.display.update()

        # Xử lý sự kiện
        if wait and not self.finished:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            self.clock.tick(FPS)
            time.sleep(0.1)

    def run(self):
        # Khởi tạo và chạy thuật toán
        self.tao_vat()
        self.update_display()

        while True:
            # Xử lý sự kiện
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            self.update_display()

            # Cập nhật thuật toán
            if self.running and not self.finished:
                prev_cost = self.c_i
                self.xu_ly_lo()

                # Hiển thị kết quả
                if self.path_new and self.c_i != prev_cost:
                    print(f"Iteration {self.so_lan_lap}: Best cost = {self.c_i:.2f}")

                if not self.running:
                    self.finished = True
                    print(
                        f"Final path cost: {self.c_i:.2f}"
                        if self.path_new
                        else "No path found to goal"
                    )

            self.clock.tick(FPS)
            time.sleep(0.1)

    def lay_mau(self):
        samples = []
        attempts = 0
        max_attempts = 1000
        while len(samples) < SIZE_LO and attempts < max_attempts:
            attempts += 1
            state = (random.randint(0, WIDTH), random.randint(0, HEIGHT))
            if self.trong_vat(state) and self.trong_elip(state):
                samples.append(Node(state))
        return samples


if __name__ == "__main__":
    BITStar().run()
