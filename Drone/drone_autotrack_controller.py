# autotrack_controller.py
import time
from typing import Tuple, Optional


class PID:
    """
    時間取樣版 PID（離散）
    - 支援固定 sample_time 或自動 dt
    - 積分限幅 anti-windup
    - 可選擇是否累積積分（在鎖定死區時可停用，避免windup）
    - 輸出限幅（可選）
    """
    def __init__(
        self,
        kp: float, ki: float, kd: float,
        sample_time: Optional[float] = None,      # 固定取樣時間（秒）。None 表示不固定，依據實際 dt。
        integral_limit: float = 1.0,              # |∫e dt| 上限
        output_limits: Optional[Tuple[float, float]] = None  # (min, max) 輸出限幅，可為 None
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        self.sample_time = sample_time if (sample_time is None or sample_time > 0) else None
        self.integral_limit = abs(float(integral_limit))
        self.output_limits = output_limits

        self.reset()

    def reset(self):
        self._i = 0.0
        self._prev_err = None
        self._last_t = None
        self._last_out = 0.0

    def update(self, err: float, dt: Optional[float] = None, *, integrate: bool = True) -> float:
        """
        err: 目標 - 目前 的誤差
        dt : 可手動提供周期（秒）；若為 None，會用 time.monotonic() 自動計算。
        integrate: 是否累積積分項（在誤差落入死區時可關閉，避免windup）
        回傳：控制輸出 u
        """
        # 1) 取得 dt
        now = time.monotonic()
        if dt is None:
            if self._last_t is None:
                self._last_t = now
                self._prev_err = err
                self._last_out = self.kp * err
                return self._last_out
            dt = now - self._last_t
            self._last_t = now

        if dt <= 0:
            return self._last_out

        if (self.sample_time is not None) and (dt < self.sample_time):
            return self._last_out

        if self.sample_time is not None:
            dt = self.sample_time

        # 2) 微分
        if self._prev_err is None:
            d = 0.0
        else:
            d = (err - self._prev_err) / dt
        self._prev_err = err

        # 3) 積分（可停用）
        if integrate:
            self._i += err * dt
            if self._i > self.integral_limit:
                self._i = self.integral_limit
            elif self._i < -self.integral_limit:
                self._i = -self.integral_limit

        # 4) PID 組合
        u = self.kp * err + self.ki * self._i + self.kd * d

        # 5) 輸出限幅
        if self.output_limits is not None:
            umin, umax = self.output_limits
            if umin is not None and u < umin:
                u = umin
            if umax is not None and u > umax:
                u = umax

        self._last_out = u
        return u


class AutoTrackController:
    """
    車子式自動追蹤控制（無側移）：
    - vy 固定為 0
    - yaw_rate 用水平誤差 dx 矯正
    - vz 用垂直誤差 dy 矯正（支援下偏置 bias）
    - vx 用框面積區間（[area_min, area_max]）決定前進/後退，帶遲滯

    正規化：
      dx = (cx - fx) / fx   （右正）
      dy = (cy - fy) / fy   （下正）
    期望對準點：(0, bias_y)
    鎖定區域（死區）：|dx| <= lock_tol_x 且 |dy - bias_y| <= lock_tol_y
    面積區間：小於 area_min → 前進；大於 area_max → 後退；區間內 vx = 0（含遲滯）
    """
    def __init__(
        self,
        # 構圖目標：偏下 12%
        bias_y: float = 0.12,
        # 中心鎖定死區（以畫面比例）
        lock_tol_x: float = 0.05,
        lock_tol_y: float = 0.05,
        # 固定尺寸鎖定與距離判斷（像素）
        lock_box: float = 100.0,   # 鎖定框邊長（視覺上理想大小）
        min_box: float = 80.0,     # 小於→前進
        max_box: float = 120.0,    # 大於→後退

        # PID 參數（kp, ki, kd, integral_limit）
        pid_yaw=(0.9, 0.0, 0.25, 0.5),     # 水平→航向角
        pid_z=(0.35, 0.0, 0.06, 0.5),     # 垂直→升降
        pid_vx=(0.02, 0.0, 0.01, 0.3),     # 前後速度（用框大小誤差）

        # 輸出限制與最小動作門檻（避免小抖動）
        clip_vx: float = 1.0,
        clip_vz: float = 0.6,
        clip_yaw: float = 0.6,
        min_vx: float = 0.05,              # 小於此值則視為 0
        min_vz: float = 0.03,
        min_yaw: float = 0.03,

        # PID 固定取樣（可填固定迴圈秒數，例如 0.02）
        pid_sample_time: Optional[float] = None,
    ):
        self.bias_y = float(bias_y)
        self.lock_tol_x = abs(float(lock_tol_x))
        self.lock_tol_y = abs(float(lock_tol_y))

        self.lock_box = float(lock_box)
        self.min_box = float(min_box)
        self.max_box = float(max_box)

        kyw, iyw, dyw, limyw = pid_yaw
        kz,  iz,  dz,  limz  = pid_z
        kv,  iv,  dv,  limv  = pid_vx

        self.pid_yaw = PID(kyw, iyw, dyw, sample_time=pid_sample_time, integral_limit=limyw)
        self.pid_z   = PID(kz,  iz,  dz,  sample_time=pid_sample_time, integral_limit=limz)
        self.pid_vx  = PID(kv,  iv,  dv,  sample_time=pid_sample_time, integral_limit=limv)

        self.clip_vx  = float(clip_vx)
        self.clip_vz  = float(clip_vz)
        self.clip_yaw = float(clip_yaw)
        self.min_vx   = abs(float(min_vx))
        self.min_vz   = abs(float(min_vz))
        self.min_yaw  = abs(float(min_yaw))

    @staticmethod
    def _clip(v: float, lo: float, hi: float) -> float:
        return lo if v < lo else (hi if v > hi else v)

    @staticmethod
    def _deadzone(v: float, dz: float) -> float:
        return 0.0 if abs(v) < dz else v

    def reset(self):
        self.pid_yaw.reset()
        self.pid_z.reset()
        self.pid_vx.reset()

    def compute(
        self,
        tlwh: Tuple[float, float, float, float],
        image_size: Tuple[int, int],
        dt: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        x, y, w, h = tlwh
        im_w, im_h = image_size
        if im_w <= 0 or im_h <= 0 or w <= 0 or h <= 0:
            return 0.0, 0.0, 0.0, 0.0

        # 影像中心
        fx = im_w * 0.5
        fy = im_h * 0.5

        # 目標中心
        cx = x + 0.5 * w
        cy = y + 0.5 * h

        # 正規化誤差（右/下為正）
        dx = (cx - fx) / fx
        dy = (cy - fy) / fy

        # 垂直目標在偏下 bias_y
        dy_err = dy - self.bias_y

        # ========== 航向角（Yaw） ==========
        in_lock_x = abs(dx) <= self.lock_tol_x
        yaw_raw = self.pid_yaw.update(dx, dt=dt, integrate=not in_lock_x)
        yaw_rate = self._clip(yaw_raw, -self.clip_yaw, self.clip_yaw)
        yaw_rate = 0.0 if in_lock_x else self._deadzone(yaw_rate, self.min_yaw)

        # ========== 升降（Vz） ==========
        in_lock_y = abs(dy_err) <= self.lock_tol_y
        vz_raw = self.pid_z.update(dy_err, dt=dt, integrate=not in_lock_y)
        vz = self._clip(vz_raw, -self.clip_vz, self.clip_vz)
        vz = 0.0 if in_lock_y else self._deadzone(vz, self.min_vz)

        # ========== 前後（Vx）：用框大小（像素） ==========
        # 用平均邊長代表距離指標；也可改 min(w,h) 或幾何平均
        box_size = 0.5 * (w + h)

        if box_size < self.min_box:          # 太小 → 前進
            err = (box_size - self.min_box) / max(self.min_box, 1.0)
            vx_raw = -self.pid_vx.update(err, dt=dt, integrate=True)
        elif box_size > self.max_box:        # 太大 → 後退
            err = (box_size - self.max_box) / max(self.max_box, 1.0)
            vx_raw = -self.pid_vx.update(err, dt=dt, integrate=True)
        else:                                 # 在 80~120 區間內 → 停積分、vx=0
            self.pid_vx.update(0.0, dt=dt, integrate=False)
            vx_raw = 0.0

        vx = self._clip(vx_raw, -self.clip_vx, self.clip_vx)
        vx = 0.0 if box_size >= self.min_box and box_size <= self.max_box else self._deadzone(vx, self.min_vx)

        # 同時滿足：中心誤差在死區、且大小在鎖定區內 → 完全靜止
        in_lock_size = (self.min_box <= box_size <= self.max_box)
        if in_lock_x and in_lock_y and in_lock_size:
            vx, vz, yaw_rate = 0.0, 0.0, 0.0

        # 車子式移動，無側移
        vy = 0.0
        return vx, vy, vz, yaw_rate
