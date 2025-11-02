import numpy as np
from ..utils.viz import draw_rectangles
def gen_sequence(num_frames=10, width=256, height=160, num_rect=5, dx=2, jitter=1, save_dir=None):
    frames = []
    base_xs = np.linspace(20, width-40, num_rect).astype(int)
    rects0 = [(int(x), 40, 20, 40) for x in base_xs]
    for t in range(num_frames):
        rects = []
        for (x,y,w,h) in rects0:
            xt = int(x + dx*t + np.random.randint(-jitter,jitter+1))
            rects.append((xt,y,w,h))
        frames.append({"rects": rects})
    seq = {"width": width, "height": height, "frames": frames, "gt": {"repeat_axis":"x", "regular": True}}
    if save_dir:
        import os; os.makedirs(save_dir, exist_ok=True)
        for t, fr in enumerate(frames):
            draw_rectangles(width, height, fr['rects'], path=f"{save_dir}/frame_{t:03d}.png")
    return seq
