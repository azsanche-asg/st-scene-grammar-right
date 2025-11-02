from typing import Dict, Any, Tuple
import numpy as np
class HeuristicPolicy:
    def __init__(self, iou_thresh: float = 0.3):
        self.iou_thresh = iou_thresh
    @staticmethod
    def iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
        ax,ay,aw,ah = a; bx,by,bw,bh = b
        x1, y1 = max(ax,bx), max(ay,by)
        x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
        inter = max(0, x2-x1) * max(0, y2-y1)
        ua = aw*ah + bw*bh - inter + 1e-6
        return inter / ua
    def induce(self, seq: Dict[str, Any]) -> Dict[str, Any]:
        tracks = []
        for t, frame in enumerate(seq['frames']):
            rects = frame['rects']
            assigned = [False]*len(rects)
            for tr in tracks:
                last = tr[-1]
                best_i, best_v = -1, 0.0
                for i,r in enumerate(rects):
                    if assigned[i]: continue
                    v = self.iou(last, r)
                    if v > best_v:
                        best_i, best_v = i, v
                if best_v >= self.iou_thresh and best_i >= 0:
                    tr.append(rects[best_i]); assigned[best_i]=True
                else:
                    tr.append(tr[-1])
            for i,r in enumerate(rects):
                if not assigned[i]:
                    tracks.append([r]*(t) + [r])
        T = len(seq['frames'])
        for tr in tracks:
            if len(tr) < T: tr += [tr[-1]]*(T-len(tr))
        xs = sorted([tr[0][0] for tr in tracks]) if tracks else []
        diffs = np.diff(xs) if len(xs)>1 else np.array([])
        rep = bool(diffs.size>0 and (diffs.std() / (diffs.mean()+1e-6) < 0.2))
        grammar = {
            "parts": [{"track": tr} for tr in tracks],
            "rules": [{"type": "repeat", "axis": "x", "approx_equal_spacing": rep}] if rep else [],
            "meta": {"policy": "heuristic", "iou_thresh": self.iou_thresh}
        }
        return grammar
