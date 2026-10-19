"""
FallbackRecognizer: Geometry-based ASL letter/number recognition.
Uses MediaPipe landmark geometry — no GPU required.

All thresholds are derived from empirical measurement on the Kaggle ASL Alphabet
training set (3000 images/class, 15 samples per letter analyzed).
Key measurements used:
  T_ext       = |thumb_tip_x - thumb_ip_x|  (lateral thumb extension)
  T_up        = thumb_mcp_y - thumb_tip_y   (thumb pointing up, positive = up)
  T_x_IMCP    = thumb_tip_x - index_mcp_x   (negative = thumb over fingers)
  T_y_IMCP    = index_mcp_y - thumb_tip_y   (positive = thumb above index MCP)
  d_T_I       = dist(thumb_tip, index_tip)
  d_T_IMCP    = dist(thumb_tip, index_mcp)
  idx_mid_x   = |index_tip_x - middle_tip_x|
"""
import numpy as np
from typing import Tuple


class FallbackRecognizer:
    """
    Data-driven geometry ASL recognizer.
    Same interface as ClassifierHead: returns (label, confidence).
    """

    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP  = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP   = 13, 14, 15, 16
    PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP  = 17, 18, 19, 20

    def __init__(self):
        self.MATCHED_CONFIDENCE  = 0.87
        self.NO_MATCH_CONFIDENCE = 0.0

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _ext(self, lm, tip, pip):
        """Finger extended: tip above (lower y) than pip."""
        return bool(lm[tip][1] < lm[pip][1])

    def _dist(self, lm, a, b):
        return float(np.linalg.norm(lm[a] - lm[b]))

    def _angle(self, v1, v2):
        c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    def _features(self, lm):
        """
        Extract all discriminating features from a 21-landmark array.
        Returns a dict of named measurements.
        """
        T, I, M, R, P = (
            self._ext(lm, self.THUMB_TIP,  self.THUMB_IP),   # lateral thumb
            self._ext(lm, self.INDEX_TIP,  self.INDEX_PIP),
            self._ext(lm, self.MIDDLE_TIP, self.MIDDLE_PIP),
            self._ext(lm, self.RING_TIP,   self.RING_PIP),
            self._ext(lm, self.PINKY_TIP,  self.PINKY_PIP),
        )
        # Thumb lateral extension magnitude
        t_ext = abs(lm[self.THUMB_TIP][0] - lm[self.THUMB_IP][0])
        # Thumb pointing up (positive = tip above MCP)
        t_up  = lm[self.THUMB_MCP][1] - lm[self.THUMB_TIP][1]
        # Thumb tip x relative to index MCP (negative = thumb over fingers)
        t_x_imcp = lm[self.THUMB_TIP][0] - lm[self.INDEX_MCP][0]
        # Thumb tip y relative to index MCP (positive = thumb above index MCP)
        t_y_imcp = lm[self.INDEX_MCP][1] - lm[self.THUMB_TIP][1]
        # Distances
        d_ti   = self._dist(lm, self.THUMB_TIP, self.INDEX_TIP)
        d_tm   = self._dist(lm, self.THUMB_TIP, self.MIDDLE_TIP)
        d_tr   = self._dist(lm, self.THUMB_TIP, self.RING_TIP)
        d_tp   = self._dist(lm, self.THUMB_TIP, self.PINKY_TIP)
        d_timcp= self._dist(lm, self.THUMB_TIP, self.INDEX_MCP)
        # Index-middle spread
        idx_mid_x = abs(lm[self.INDEX_TIP][0] - lm[self.MIDDLE_TIP][0])
        # Index direction
        iv = lm[self.INDEX_TIP] - lm[self.INDEX_MCP]
        idx_horiz = abs(iv[0]) > abs(iv[1])
        idx_down  = iv[1] > 0.02   # tip below MCP
        # Index curl depth (positive = curled)
        i_curl = lm[self.INDEX_TIP][1] - lm[self.INDEX_PIP][1]
        # Index tip-pip y gap (small = hooked/barely extended)
        i_tip_pip_y = abs(lm[self.INDEX_TIP][1] - lm[self.INDEX_PIP][1])

        return dict(
            T=T, I=I, M=M, R=R, P=P,
            t_ext=t_ext, t_up=t_up,
            t_x_imcp=t_x_imcp, t_y_imcp=t_y_imcp,
            d_ti=d_ti, d_tm=d_tm, d_tr=d_tr, d_tp=d_tp, d_timcp=d_timcp,
            idx_mid_x=idx_mid_x,
            idx_horiz=idx_horiz, idx_down=idx_down,
            i_curl=i_curl, i_tip_pip_y=i_tip_pip_y,
        )

    def _all_curled(self, lm):
        return (
            not self._ext(lm, self.INDEX_TIP,  self.INDEX_PIP)  and
            not self._ext(lm, self.MIDDLE_TIP, self.MIDDLE_PIP) and
            not self._ext(lm, self.RING_TIP,   self.RING_PIP)   and
            not self._ext(lm, self.PINKY_TIP,  self.PINKY_PIP)
        )

    # ── Main classifier ───────────────────────────────────────────────────────

    def _classify(self, lm: np.ndarray) -> str | None:
        f = self._features(lm)
        T, I, M, R, P = f['T'], f['I'], f['M'], f['R'], f['P']

        # ── O: tight circle — check FIRST (before C which is looser) ─────────
        # Empirical: d_T_I=0.038, idx_mid_x=0.015, all fingers curved to thumb
        # O has thumb-index distance VERY small and fingers bunched together
        if (f['d_ti'] < 0.07 and f['idx_mid_x'] < 0.06 and
                not I and not M):
            return 'O'

        # ── C: curved C-shape ─────────────────────────────────────────────────
        # C has fingers partially curved (not fully extended, not fully curled)
        # All fingers at similar height, thumb not extended
        # Must fire BEFORE B/5/W/F blocks
        tip_y  = lm[self.INDEX_TIP][1]
        pip_y  = lm[self.INDEX_PIP][1]
        mcp_y  = lm[self.INDEX_MCP][1]
        mid_tip_y  = lm[self.MIDDLE_TIP][1]
        mid_pip_y  = lm[self.MIDDLE_PIP][1]
        mid_mcp_y  = lm[self.MIDDLE_MCP][1]
        ring_tip_y = lm[self.RING_TIP][1]
        ring_pip_y = lm[self.RING_PIP][1]
        ring_mcp_y = lm[self.RING_MCP][1]
        # C: tips are between MCP and PIP height (partially curved)
        # tip_y > pip_y - 0.02 means tip is at or below pip (curved)
        # tip_y < mcp_y + 0.05 means tip is above MCP (not fully curled)
        c_idx  = (tip_y > pip_y - 0.02) and (tip_y < mcp_y + 0.05)
        c_mid  = (mid_tip_y > mid_pip_y - 0.02) and (mid_tip_y < mid_mcp_y + 0.05)
        c_ring = (ring_tip_y > ring_pip_y - 0.02) and (ring_tip_y < ring_mcp_y + 0.05)
        if c_idx and c_mid and c_ring and f['t_ext'] < 0.07 and f['d_ti'] > 0.08:
            return 'C'

        # ── F: thumb-index pinch, middle+ring+pinky extended ─────────────────
        # Empirical: T_ext=0.015 (NOT ext), d_T_I=0.056 (VERY close)
        # F has index curled into pinch — index NOT extended
        if M and R and P and f['d_ti'] < 0.09 and not I:
            return 'F'

        # B: all 4 fingers extended, thumb tucked (d_ti large)
        # Empirical: T_ext=0.061, d_T_I=0.284
        if I and M and R and P:
            if f['t_ext'] < 0.09 and f['d_ti'] > 0.20:
                return 'B'
            return '5'

        # W: index+middle+ring extended, pinky down
        if I and M and R and not P:
            return 'W'

        # ── Letters with index+middle extended ───────────────────────────────

        if I and M and not R and not P:
            # R: index+middle very close (crossed)
            if f['idx_mid_x'] < 0.03:
                return 'R'
            # H: index+middle horizontal
            if f['idx_horiz']:
                return 'H'
            # K/V/U disambiguation using thumb + spread:
            # K: thumb extended (T=True), moderate spread
            # V: thumb NOT extended, large spread
            # U: thumb NOT extended, small spread
            # Empirical: K d_timcp=0.085, V d_timcp=0.129, U d_timcp=0.133
            if T and f['t_ext'] > 0.05:
                # Thumb is clearly extended → K
                return 'K'
            # No thumb: V vs U by spread
            if f['idx_mid_x'] >= 0.08:
                return 'V'
            return 'U'

        # ── Letters with only index extended ─────────────────────────────────

        if I and not M and not R and not P:
            # T: index barely extended (borderline), thumb pointing UP high
            # Empirical T: t_up=0.234, t_x_imcp=+0.207, i_tip_pip_y=0.007 (barely ext)
            # Catch T here before Z/G/D/L/X
            if f['t_up'] > 0.15 and f['t_x_imcp'] > 0.10 and f['i_tip_pip_y'] < 0.04:
                return 'T'
            # Z: index pointing far sideways
            # Empirical: idx_mid_x=0.276
            if f['idx_mid_x'] > 0.20:
                return 'Z'
            # G: index pointing sideways (horizontal)
            # Empirical: idx_mid_x=0.235, idx_horiz=True
            if f['idx_horiz'] and f['idx_mid_x'] > 0.10:
                return 'G'
            # D: index extended, thumb touches middle — check BEFORE L/X/Z
            # Empirical: d_T_M=0.069
            if f['d_tm'] < 0.09:
                return 'D'
            # L: index + thumb at ~90°
            if T and f['t_ext'] > 0.05:
                tv = lm[self.THUMB_TIP] - lm[self.THUMB_CMC]
                iv = lm[self.INDEX_TIP] - lm[self.INDEX_MCP]
                angle = self._angle(tv[:2], iv[:2])
                if 50 < angle < 130:
                    return 'L'
            # P: index pointing DOWN, thumb out and below
            if f['idx_down'] and f['t_y_imcp'] < -0.15:
                return 'P'
            # Q: index+thumb both pointing down
            if f['t_y_imcp'] < -0.18:
                return 'Q'
            # X: index barely extended (hooked), thumb out
            # Empirical X: i_tip_pip_y=0.074 (small), T_ext=0.082
            if T and f['i_tip_pip_y'] < 0.10:
                return 'X'
            # Default single-index → Z
            return 'Z'

        # ── Letters with only pinky extended ─────────────────────────────────

        if not I and not M and not R and P:
            # Y: pinky + thumb, thumb beside fist (NOT over fingers)
            # Empirical Y: T_x_IMCP=+0.163 (thumb beside/outside)
            # Empirical I: T_x_IMCP=-0.106 (thumb over palm)
            if f['t_x_imcp'] > 0.05:
                return 'Y'
            return 'I'

        # ── Fist-based letters (all 4 fingers curled) ────────────────────────
        if self._all_curled(lm):
            # Empirical measurements:
            # A:  t_y_imcp=+0.112 (thumb ABOVE I_MCP), t_x_imcp=+0.053 (beside)
            # E:  t_x_imcp=-0.140 (thumb OVER), d_ti=0.147
            # S:  t_x_imcp=-0.098 (thumb over), d_ti=0.101
            # T:  t_up=0.234 (HIGH), t_x_imcp=+0.207 (beside, pointing up)
            # M:  t_y_imcp=-0.156 (thumb BELOW I_MCP)
            # N:  t_y_imcp=-0.193 (even lower)
            # Q:  t_y_imcp=-0.251 (lowest)

            # Q: thumb way below index MCP
            # Use stricter threshold to avoid eating M/N
            if f['t_y_imcp'] < -0.22:
                return 'Q'

            # M/N: thumb below index MCP
            if f['t_y_imcp'] < -0.10:
                if f['t_y_imcp'] < -0.17:
                    return 'N'
                return 'M'

            # T: thumb pointing strongly UP and beside fingers
            # Empirical: t_up=0.234, t_x_imcp=+0.207
            # Use lower threshold to catch more T cases
            if f['t_up'] > 0.12 and f['t_x_imcp'] > 0.08:
                return 'T'

            # E vs S: both have thumb over fingers (t_x_imcp < 0)
            # E: d_ti=0.147 (thumb farther from index tip)
            # S: d_ti=0.101 (thumb wraps closer over index)
            if f['t_x_imcp'] < -0.05:
                if f['d_ti'] > 0.12:
                    return 'E'
                return 'S'

            # A: thumb beside fist, above index MCP
            return 'A'

        # ── NUMBERS (fallback after all letters) ──────────────────────────────
        if I and M and R and P and T:   return '5'
        if I and M and R and P:         return '4'
        if I and M and R:               return '3'
        if I and M:                     return '2'
        if I:                           return '1'
        if not I and not M and not R and P and T: return '6'
        if not I and not M and not R and not P and T: return '7'
        if not I and M and R and P:     return '8'
        if not I and not M and R and P: return '9'

        return None

    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        landmarks: (21, 3) normalized landmark array from one hand.
        Returns (label, confidence).
        """
        if landmarks is None or landmarks.shape[0] < 21:
            return ('', self.NO_MATCH_CONFIDENCE)
        label = self._classify(landmarks)
        if label is not None:
            return (label, self.MATCHED_CONFIDENCE)
        return ('', self.NO_MATCH_CONFIDENCE)
