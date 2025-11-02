import numpy as np
def raster_repeat_score(images):
    scores = []
    for im in images:
        col_mean = im.mean(axis=0).mean(axis=1)
        scores.append(float(col_mean.var()))
    return float(np.mean(scores)) if scores else 0.0
